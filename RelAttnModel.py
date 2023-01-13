from torch import nn
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import BartConfig
from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartDecoder,
    BartDecoderLayer,
    _expand_mask,
    BartModel,
    shift_tokens_right,
    BartForConditionalGeneration,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.activations import ACT2FN
import random
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
import pdb
from math import pi, sqrt, exp, log
import numpy as np
import torch.nn.functional as F

logger = logging.get_logger(__name__)


class BartRelConfig(BartConfig):
    def __init__(
        self,
        rel_attn_weight: float = 0,
        fixed_rel_attn_weight: bool = True,
        rel_attn_type: str = "fixed",
        smooth_method: str = None,
        smooth_window: int = 0,
        smooth_gaussian_sigma: float = 1,
        rel_attn_weight_perhead: bool = False,
        rel_attn_weight_linear: bool=False,
        rel_attn_weight_with_ca_embed: bool=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rel_attn_weight = rel_attn_weight
        self.fixed_rel_attn_weight = fixed_rel_attn_weight
        self.rel_attn_type = rel_attn_type
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window
        self.smooth_gaussian_sigma = smooth_gaussian_sigma
        self.rel_attn_weight_perhead = rel_attn_weight_perhead
        self.rel_attn_weight_linear=rel_attn_weight_linear
        self.rel_attn_weight_with_ca_embed=rel_attn_weight_with_ca_embed


class RelSeq2SeqModelOutput(Seq2SeqModelOutput):
    def __init__(self, rel_attn: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.rel_attn = rel_attn


class RelSeq2SeqLMOutput(Seq2SeqLMOutput):
    def __init__(self, rel_attn: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.rel_attn = rel_attn


class BartRelAttention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        rel_attn_weight: float = 0.5,
        fixed_rel_attn_weight: bool = True,
        rel_attn_weight_perhead: bool = False,
        rel_attn_weight_linear: bool = False,
        rel_attn_weight_with_ca_embed: bool=False
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.fixed_rel_attn_weight = fixed_rel_attn_weight
        if fixed_rel_attn_weight:
            self.rel_attn_weight_type = 'fixed'
            # self.rel_attn_weight = torch.Tensor([rel_attn_weight])
            self.rel_attn_weight = nn.Parameter(
                data=torch.log(torch.Tensor([rel_attn_weight/(1-rel_attn_weight)])), requires_grad=False
            )
        else:
            if rel_attn_weight_linear:
                self.rel_attn_weight_type = 'linear'
                self.rel_attn_weight_with_ca_embed =rel_attn_weight_with_ca_embed
                if rel_attn_weight_with_ca_embed:
                    self.rel_attn_weight =  nn.Sequential(
                                                nn.Linear(2*embed_dim,num_heads),
                                                nn.Sigmoid()
                                            )
                else:
                    self.rel_attn_weight =  nn.Sequential(
                                                nn.Linear(embed_dim,num_heads),
                                                nn.Sigmoid()
                                            )
                self.rel_attn_weight[0].weight.data.zero_()
                self.rel_attn_weight[0].bias.data.fill_(log(rel_attn_weight/(1-rel_attn_weight)))
            elif rel_attn_weight_perhead:
                self.rel_attn_weight_type = 'param_perhead'
                self.rel_attn_weight = nn.Parameter(
                    data=torch.log(
                        torch.Tensor([rel_attn_weight/(1-rel_attn_weight) for _ in range(num_heads)])
                    ),
                    requires_grad=True,
                )
            else:
                self.rel_attn_weight_type = 'param_perlayer'
                self.rel_attn_weight = nn.Parameter(
                    data=torch.log(torch.Tensor([rel_attn_weight/(1-rel_attn_weight)])), requires_grad=True
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_attn: Optional[torch.Tensor] = None,
        control_embedding=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if rel_attn is not None:
            if self.rel_attn_weight_type=='param_perlayer' or self.rel_attn_weight_type=='fixed':
                num_weights=1
            else:
                num_weights=self.num_heads
            rel_attn = rel_attn.view(bsz, -1, 1, src_len).repeat(
                1, num_weights, 1, 1
            )
            # if self.fixed_rel_attn_weight:
            #     rel_attn_weight = self.rel_attn_weight.view(1, -1, 1, 1)
            # else:
            if self.rel_attn_weight_type=='linear':
                if self.rel_attn_weight_with_ca_embed:
                    doc_control_embedding = torch.concat([torch.mean(control_embedding,1),torch.mean(key_value_states,1)],dim=-1)
                else:
                    doc_control_embedding= torch.mean(key_value_states,1)
                # doc_control_embedding= torch.mean(key_value_states,1)
                rel_attn_weight = self.rel_attn_weight(doc_control_embedding).view(bsz,self.num_heads,1,1)
            else:
                rel_attn_weight = torch.sigmoid(self.rel_attn_weight).view(1, -1, 1, 1)
            attn_weights = (1 - rel_attn_weight) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ) + rel_attn_weight * rel_attn

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartRelDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BartRelConfig):
        super().__init__(config)
        self.encoder_attn = BartRelAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            rel_attn_weight=config.rel_attn_weight,
            fixed_rel_attn_weight=config.fixed_rel_attn_weight,
            rel_attn_weight_perhead=config.rel_attn_weight_perhead,
            rel_attn_weight_linear = config.rel_attn_weight_linear,
            rel_attn_weight_with_ca_embed=config.rel_attn_weight_with_ca_embed
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rel_attn: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        control_embedding=None
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                rel_attn=rel_attn,
                control_embedding=control_embedding
            )

            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartRelDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self, config: BartRelConfig, embed_tokens: Optional[nn.Embedding] = None
    ):
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [BartRelDecoderLayer(config) for _ in range(config.decoder_layers)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rel_attn=None,
        control_embedding=None
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                    rel_attn,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    rel_attn=rel_attn,
                    control_embedding=control_embedding
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartRelModel(BartModel):
    def __init__(self, config: BartRelConfig):
        super().__init__(config)
        self.decoder = BartRelDecoder(config, self.shared)
        self.init_weights()
        self.rel_attn_type = config.rel_attn_type
        self.smooth_method = config.smooth_method
        if self.smooth_method == "Gaussian":
            self.smooth_window = config.smooth_window
            self.smooth_gaussian_sigma = nn.Parameter(
                data=torch.Tensor([config.smooth_gaussian_sigma]), requires_grad=False
            )
        if config.rel_attn_type == "trained":
            self.rel_k_proj = nn.Linear(config.d_model, config.d_model)
            self.rel_q_proj = nn.Linear(config.d_model, config.d_model)
        # ## zero-shot
        # self.rel_k_proj.weight.data = torch.ones_like(self.rel_k_proj.weight)
        # self.rel_k_proj.weight.requires_grad = False
        # self.rel_k_proj.bias = None
        # self.rel_q_proj.weight.data = torch.ones_like(self.rel_q_proj.weight)
        # self.rel_q_proj.weight.requires_grad = False
        # self.rel_q_proj.bias = None

    def get_gaussian_filter(self, half_window_size, sigma=torch.FloatTensor([1])):
        n = 2 * half_window_size + 1
        r = torch.tensor(range(-int(n / 2), int(n / 2) + 1), device=sigma.device)
        sigma = sigma.repeat(n)
        filter = 1 / (sigma * sqrt(2 * pi)) * torch.exp(-(r ** 2) / (2 * sigma ** 2))
        return filter
        # np.array(
        #     [
        #         1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2))
        #         for x in r
        #     ]
        # )

    def get_rel_attn(
        self,
        controll_embedding,
        input_embedding,
        attention_mask=None,
        control_aspect_mask=None,
    ):
        # batch*k*dim
        # controll_embedding = self.shared(control_aspect_ids)
        bsz, k, _ = controll_embedding.shape
        _, src_len, _ = input_embedding.shape
        if self.rel_attn_type == "fixed":
            q_state = controll_embedding
            k_state = input_embedding
        else:
            q_state = self.rel_q_proj(controll_embedding)
            k_state = self.rel_k_proj(input_embedding)
        attn_weights = torch.bmm(q_state, k_state.transpose(1, 2))
        if control_aspect_mask is not None:
            attn_weights = attn_weights.view(
                bsz, k, src_len
            ) * control_aspect_mask.view(bsz, k, 1)
        attn_weights = attn_weights.mean(dim=1).view(bsz, src_len)
        if self.smooth_method == "Gaussian":
            gaussian_filter = self.get_gaussian_filter(
                self.smooth_window, self.smooth_gaussian_sigma
            )
            attn_weights = F.conv1d(
                attn_weights.view(bsz, 1, src_len),
                gaussian_filter.view(1, 1, -1),
                padding="same",
            ).squeeze(1)
        if attention_mask is not None:
            attention_mask = _expand_mask(
                attention_mask, input_embedding.dtype, tgt_len=1
            )
            attn_weights = attn_weights.view(bsz, src_len) + attention_mask.view(
                bsz, src_len
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return attn_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rel_control_aspect_ids=None,
        rel_control_aspect_mask=None,
        rel_attn=None,
        control_embedding=None
    ):
        """
        Args:
            control_aspect_ids: bsz*k
            control_aspect_mask: bsz*k
        """
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if rel_attn is None:
            if inputs_embeds is None:
                inputs_embeds = encoder_outputs[1][0]
            control_embedding = self.shared(rel_control_aspect_ids)
            rel_attn = self.get_rel_attn(
                control_embedding,
                inputs_embeds,
                attention_mask=attention_mask,
                control_aspect_mask=rel_control_aspect_mask,
            )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rel_attn=rel_attn,
            control_embedding=control_embedding
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return RelSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            rel_attn=rel_attn,
        )


class BartRelForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartRelConfig):
        super().__init__(config)
        self.model = BartRelModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rel_control_aspect_ids=None,
        rel_control_aspect_mask=None,
        rel_attn=None,
        control_embedding=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rel_control_aspect_ids=rel_control_aspect_ids,
            rel_control_aspect_mask=rel_control_aspect_mask,
            rel_attn=rel_attn,
            control_embedding=control_embedding
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return RelSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            rel_attn=outputs.rel_attn,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        rel_control_aspect_ids=None,
        rel_control_aspect_mask=None,
        rel_attn=None,
        control_embedding=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "rel_control_aspect_ids": rel_control_aspect_ids,
            "rel_control_aspect_mask": rel_control_aspect_mask,
            "rel_attn": rel_attn,
            "control_embedding":control_embedding
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_")
                    or argument.startswith("cross_attn")
                    or argument.startswith("rel_")
                )
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(
                input_ids, return_dict=True, **encoder_kwargs
            )
        if "rel_attn" not in model_kwargs and "rel_control_aspect_ids" in model_kwargs:

            if "inputs_embeds" not in model_kwargs:
                inputs_embeds = model_kwargs["encoder_outputs"][1][0]
            control_embedding = self.model.shared(model_kwargs["rel_control_aspect_ids"])
            rel_attn = self.model.get_rel_attn(
                control_embedding,
                inputs_embeds,
                attention_mask=model_kwargs["attention_mask"],
                control_aspect_mask=model_kwargs["rel_control_aspect_ids"]
                if "rel_control_aspect_ids" in model_kwargs
                else None,
            )
            model_kwargs["rel_attn"] = rel_attn
            model_kwargs["control_embedding"]=control_embedding
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        if "rel_attn" in model_kwargs:
            rel_attn = model_kwargs["rel_attn"]
            model_kwargs["rel_attn"] = rel_attn.index_select(0, expanded_return_idx)
        if "control_embedding" in model_kwargs:
            control_embedding=model_kwargs["control_embedding"]
            model_kwargs["control_embedding"] = control_embedding.index_select(0, expanded_return_idx)
        return input_ids, model_kwargs
