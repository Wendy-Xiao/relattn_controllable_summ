from transformers import  PegasusForConditionalGeneration
from transformers import  BartForConditionalGeneration
from transformers import PreTrainedModel,AutoModelForSeq2SeqLM
from typing import List, Optional, Tuple, Union
from typing import Any, Dict
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn.functional as F
import pdb

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10


class PPLM(PreTrainedModel):
    def __init__(self,config, pretrained_name, cache_dir, pplm_step_size,gm_scale=0.95,model_type='bart'):
        super().__init__(config)
        if model_type=='bart':
            self.model = BartForConditionalGeneration.from_pretrained(pretrained_name,cache_dir=cache_dir,output_hidden_states=True,
                            return_dict=True)
        elif model_type=='pegasus':
            self.model = PegasusForConditionalGeneration.from_pretrained(pretrained_name,cache_dir=cache_dir,output_hidden_states=True,
                return_dict=True)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name,cache_dir=cache_dir,output_hidden_states=True,
                return_dict=True)
        self.step_size=pplm_step_size
        self.gm_scale=gm_scale
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()
    def forward(
        self,
        output_so_far,
        bow_vector_batch,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        torch.set_grad_enabled(True)
        outputs =self.model(input_ids,
                            attention_mask,
                            decoder_input_ids,
                            decoder_attention_mask,
                            head_mask,
                            decoder_head_mask,
                            cross_attn_head_mask,
                            encoder_outputs,
                            past_key_values,
                            inputs_embeds,
                            decoder_inputs_embeds,
                            labels,
                            use_cache=True,
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True)
        # if output_so_far.shape[1]!=1:
        pert_logits,pert_past = self.pplm_perturb(outputs,output_so_far,bow_vector_batch,encoder_outputs)
        # else:
        #     pert_logits = outputs.logits
        #     pert_past= outputs.past_key_values
        # pert_logits = outputs.logits
        # pert_past= outputs.past_key_values    

        return Seq2SeqLMOutput(
            loss=None,
            logits=pert_logits,
            past_key_values=pert_past,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def pplm_perturb(self,output,output_so_far,bow_vector_batch,encoder_outputs,
                        loss_type=1,
                        length=100,
                        temperature=1.0,
                        top_k=10,
                        sample=True,
                        num_iterations=3,
                        grad_length=10000,
                        horizon_length=1,
                        window_length=0,
                        decay=False,
                        gamma=1.5,
                        kl_scale=0.01,):
        last = output_so_far[:, -1:] #define output_so_far as the previous generated text.
        past = output.past_key_values


        unpert_logits = output.logits
        unpert_past = output.past_key_values
        unpert_all_hidden = output.decoder_hidden_states
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if output_so_far.shape[-1] >= grad_length:
            current_stepsize = self.step_size * 0
        else:
            current_stepsize = self.step_size
        # modify the past if necessary
        accumulated_hidden = unpert_last_hidden[:, :-1, :]
        accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
        (
            pert_past,
            _,
            grad_norms,
        ) = perturb_past(
            past,
            self.model,
            last,
            encoder_outputs,
            unpert_past=unpert_past,
            unpert_logits=unpert_logits,
            accumulated_hidden=accumulated_hidden,
            grad_norms=None, #### may change later to grab from prev iter
            stepsize=current_stepsize,
            bow_vector_batch=bow_vector_batch,
            loss_type=loss_type,
            num_iterations=num_iterations,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            kl_scale=kl_scale,
            device=self.device,
        )

        output = self.model(
            decoder_input_ids=last,
            past_key_values=pert_past,
            encoder_outputs=encoder_outputs,
        )
        pert_logits = output.logits
        past = output.past_key_values


        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        unpert_logits[:,-1,:] = (pert_logits * self.gm_scale) + (
            unpert_logits[:, -1, :] * (1 - self.gm_scale)
        )  # + SMALL_CONST
        return unpert_logits,past

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
        bow_vector_batch=None,
        **kwargs
    ):
        decoder_input_ids_last = decoder_input_ids
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids_last = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids_last,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "output_so_far":decoder_input_ids,
            "bow_vector_batch":bow_vector_batch
        }

    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)
        if "bow_vector_batch" in model_kwargs:
            bow_vector_batch=model_kwargs["bow_vector_batch"]
            model_kwargs["bow_vector_batch"] = bow_vector_batch.index_select(0, expanded_return_idx)
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn") or argument=='bow_vector_batch')
            }
            model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        #copied from BARTFORCONDITIONALGENERATION
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

def add_past(a, b):
    """a and b are in the same shape with past_key_value in bart output,
    i.e. tuple(tuple(tensor)), where the first tuple is of size n_layers,
    and the second tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head))
    and 2 tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head)
    """
    new_past = []
    for i in range(len(b)):
        new_past_layer = []
        for j in range(4):
            new_past_layer.append(a[i][j] + b[i][j])
        new_past.append(new_past_layer)
    return tuple([tuple(item) for item in new_past])

def build_bows_one_hot_vectors_batch(batch_bow_indices, tokenizer):
    if batch_bow_indices is None:
        return None
    ### changed to tokenizer.vocab_size-1, as the dimensions are mismatched between tokenizer and model
    # model: 50264
    # tokenizer: 50265
    if ('bart' in tokenizer.name_or_path) or ('ctrlsum' in tokenizer.name_or_path):
        bows_vector_batch = torch.zeros(
            len(batch_bow_indices), tokenizer.vocab_size - 1,
        )
    else:
        bows_vector_batch = torch.zeros(
            len(batch_bow_indices), tokenizer.vocab_size,
        )
    for i_batch, single_bow in enumerate(batch_bow_indices):
        # ??? why filter out all the words with more than one token?
        # single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        bows_vector_batch[i_batch, single_bow] = 1

    # pdb.set_trace()
    return bows_vector_batch

def build_bows_one_hot_vectors(bow_indices, vocab_size,tokenizer_name):
    if bow_indices is None:
        return None
    ### changed to tokenizer.vocab_size-1, as the dimensions are mismatched between tokenizer and model
    # model: 50264
    # tokenizer: 50265
    # pdb.set_trace()
    if ('bart' in tokenizer_name) or ('ctrlsum' in tokenizer_name):
        bows_vector = torch.zeros(vocab_size - 1 )
    else:
        bows_vector = torch.zeros(vocab_size)
    # ??? why filter out all the words with more than one token?
    # single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
    bows_vector[bow_indices] = 1

    # pdb.set_trace()
    return bows_vector

def get_bag_of_words_indices_batch(
    bow_words_batch, tokenizer, extractiveness=False, input_documents=None
) -> List[List[List[int]]]:
    # print("BOW ALL: ", bag_of_words_ids_or_paths)
    bow_batch = []
    for i_batch, words in enumerate(bow_words_batch):
        words = words.split(",")
        bow_indices = [
            tokenizer.encode(
                word.strip(), add_prefix_space=True, add_special_tokens=False
            )
            for word in words
        ]
        if extractiveness:
            bow_indices.append(input_documents[i_batch][1:-1])
        bow_batch.append([tid for wid in bow_indices for tid in wid])
    # pdb.set_trace()
    return bow_batch

def get_bag_of_words_indices(
    bow_words, tokenizer, extractiveness=False, input_document=None
) -> List[List[List[int]]]:
    words = bow_words.split(",")
    bow_indices = [
        tokenizer.encode(
            word.strip(), add_prefix_space=True, add_special_tokens=False
        )
        for word in words
    ]
    if extractiveness:
        bow_indices.append(input_document[1:-1])
    bow=[tid for wid in bow_indices for tid in wid]
# pdb.set_trace()
    return bow

def perturb_past(
    past,
    model,
    last,
    encoder_outputs,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    bow_vector_batch=None,
    classifier=None,
    class_label=None,
    loss_type=1,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):
    # Generate inital perturbed past
    ## TODO: change grad accumulator to adapt for bart in new transformers version
    # Original: past: list of tensors, with length of n_layers, each tensor is with shape (2, batch_size, num_heads, sequence_length, embed_size_per_head)
    # New:  (tuple(tuple(torch.FloatTensor)), Tuple of tuple(torch.FloatTensor) of length config.n_layers,
    # # with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) - decoder attention need to update
    # and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head) - cross attention, update?
    # changed now is the same shape as past, but with type list of list
    # grad_accumulator = [
    #     [(np.zeros(t.shape).astype("float32")) for t in p] for p in past
    # ]
    grad_accumulator = [
        [(torch.zeros(t.shape, requires_grad=True, device=device)) for t in p]
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window

    _, _, curr_length, _ = past[0][0].shape  # changed

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
            tuple(past[0][0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0][0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0][0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0][0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 3, 2)
        ones_mask = ones_mask.permute(0, 1, 3, 2)
        decoder_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2)

        window_mask = [
            decoder_mask.to(device),
            decoder_mask.clone().to(device),
            torch.ones_like(past[0][2]).to(device),
            torch.ones_like(past[0][2]).to(device),
        ]
    else:
        window_mask = [
            torch.ones_like(past[0][0]).to(device),
            torch.ones_like(past[0][1]).to(device),
            torch.ones_like(past[0][2]).to(device),
            torch.ones_like(past[0][3]).to(device),
        ]

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    # removing past from the graph
    # new_past = []
    # for p_ in past:
    #     new_p_ = []
    #     for t_ in p_:
    #         new_p_.append(t_.detach())
    #     new_past.append(tuple(new_p_))
    # past = tuple(new_past)

    for i in range(num_iterations):
        # changed to list of list
        # curr_perturbation = [
        #     [
        #         to_var(torch.from_numpy(t_), requires_grad=True, device=device)
        #         for t_ in p_
        #     ]
        #     for p_ in grad_accumulator
        # ]
        # curr_perturbation = [
        #     [torch.tensor(t_, requires_grad=True, device=device) for t_ in p_]
        #     for p_ in grad_accumulator
        # ]
        curr_perturbation = [
            [t_.detach().requires_grad_() for t_ in p_] for p_ in grad_accumulator
        ]
        # pdb.set_trace()
        # Compute hidden using perturbed past
        # perturbed_past = list(map(add, past, curr_perturbation))
        perturbed_past = add_past(past, curr_perturbation)

        # perturbed_past = past
        # changed
        _, _, curr_length, _ = curr_perturbation[0][0].shape

        output = model(
            decoder_input_ids=last,
            past_key_values=perturbed_past,
            encoder_outputs=encoder_outputs,
        )


        all_logits = output.logits
        all_hidden = output.decoder_hidden_states
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            # pdb.set_trace()
            # bow_logits = torch.mm(probs, torch.t(bow_vector_batch))
            bow_logits = probs * bow_vector_batch
            bow_loss = -torch.log(torch.sum(bow_logits))
            loss += bow_loss
            loss_list.append(bow_loss)
            # for one_hot_bow in one_hot_bows_vectors:
            #     bow_logits = torch.mm(probs, torch.t(one_hot_bow))
            #     bow_loss = -torch.log(torch.sum(bow_logits))
            #     loss += bow_loss
            #     loss_list.append(bow_loss)
            # pdb.set_trace()
        # TODO: Now I leave the discriminator parts unchanged, as we will run the baseline model only on bow
        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                output = model(
                    past=curr_unpert_past,
                    decoder_inputs_embeds=inputs_embeds,
                    encoder_outputs=encoder_outputs,
                )

                curr_unpert_past = output.past_key_values
                curr_all_hidden = output.decoder_hidden_states
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1
                )

            prediction = classifier(
                new_accumulated_hidden / (curr_length + 1 + horizon_length)
            )

            label = torch.tensor(
                prediction.shape[0] * [class_label], device=device, dtype=torch.long
            )
            discrim_loss = ce_loss(prediction, label)

            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                unpert_probs
                + SMALL_CONST
                * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = (
                SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            )
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())

        # compute gradients
        loss.backward()


        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                [
                    torch.max(
                        grad_norms[index][j], torch.norm(t_.grad * window_mask[j])
                    )
                    for j, t_ in enumerate(p_)
                ]
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                [
                    (torch.norm(t_.grad * window_mask[j]) + SMALL_CONST)
                    for j, t_ in enumerate(p_)
                ]
                for index, p_ in enumerate(curr_perturbation)
            ]
        # normalize gradients
        # grad = [
        #     [
        #         -stepsize
        #         * (t_.grad * window_mask[j] / grad_norms[index][j] ** gamma)
        #         .data.cpu()
        #         .numpy()
        #         for j, t_ in enumerate(p_)
        #     ]
        #     for index, p_ in enumerate(curr_perturbation)
        # ]
        grad = [
            [
                -stepsize * (t_.grad * window_mask[j] / grad_norms[index][j] ** gamma)
                for j, t_ in enumerate(p_)
            ]
            for index, p_ in enumerate(curr_perturbation)
        ]
        # accumulate gradient
        # grad_accumulator = list(map(add, grad, grad_accumulator))
        grad_accumulator = add_past(grad, grad_accumulator)

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            for t_ in p_:
                t_.grad.data.zero_()
        for p_ in grad_accumulator:
            for t_ in p_:
                if t_.grad:
                    t_.grad.data.zero_()

        # removing past from the graph

        new_past = []
        for p_ in past:
            new_p_ = []
            for t_ in p_:
                new_p_.append(t_.detach())
            new_past.append(tuple(new_p_))
        past = tuple(new_past)

    # apply the accumulated perturbations to the past
    # grad_accumulator = [
    #     [torch.tensor(t_, requires_grad=True, device=device) for t_ in p_]
    #     for p_ in grad_accumulator
    # ]
    # pert_past = list(map(add, past, grad_accumulator))
    pert_past = add_past(past, grad_accumulator)
    new_pert_past = []
    for p_ in pert_past:
        new_p_ = []
        for t_ in p_:
            new_p_.append(t_.detach())
        new_pert_past.append(tuple(new_p_))
    pert_past = tuple(new_pert_past)
    return (
        pert_past,
        new_accumulated_hidden,
        grad_norms,
    )


