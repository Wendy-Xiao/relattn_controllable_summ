from pathlib import Path
import re
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import argparse
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer
from transformers import (
    Adafactor,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from RelAttnModel import BartRelForConditionalGeneration, BartRelConfig
from pegasusRelAttnModel import PegasusRelForConditionalGeneration, PegasusRelConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, load_metric, Dataset
from transformers import pipeline
from tqdm import tqdm
import pdb
import numpy as np
import torch
import nltk
import os
import pandas as pd
import json
import random
from PPLM import PPLM,build_bows_one_hot_vectors,get_bag_of_words_indices
from transformers.modeling_utils import no_init_weights
from math import log

no_init_weights(True)
def preprocess_function(
    example,
    tokenizer,
    document_key,
    entity_key,
    summary_key,
    id_key,
    with_ent=True,
    max_length_input=1024,
    max_length_tgt=1024,
    return_ent_ids=False,
    use_pplm=False
):  
    #remove duplictes
    example[entity_key] = list(set(example[entity_key]))

    if with_ent:
        # entity + doc
        inputs = " ".join(example[entity_key]) + " => " + example[document_key]
    else:
        inputs = example[document_key]


    model_inputs = tokenizer(
        inputs, max_length=max_length_input, padding=True, truncation=True
    )
    if return_ent_ids:
        if 'bart' in tokenizer.name_or_path:
            control_aspect_ids = [
                tokenizer.encode(
                    ent.strip(),  add_prefix_space=True,add_special_tokens=False
                )
                for ent in example[entity_key]
            ]

        else:
            control_aspect_ids = [
                tokenizer.encode(
                    ent.strip(), add_special_tokens=False
                )
                for ent in example[entity_key]
            ]
        # flatten the entity representations
        control_aspect_ids = [i for ent in control_aspect_ids for i in ent]
        if len(control_aspect_ids) == 0:
            control_aspect_ids.append(tokenizer.pad_token_id)
    summary = example[summary_key]
    labels = tokenizer(summary, max_length=max_length_tgt, truncation=True)
    output = {}
    output["output_ids"] = labels["input_ids"]
    output["input_ids"] = model_inputs["input_ids"]
    output["reference"] = summary
    output["doc_id"] = example[id_key]
    output["entity"] = example[entity_key]
    if return_ent_ids:
        output["control_aspect_ids"] = control_aspect_ids
    if use_pplm:
        bow = ",".join(example[entity_key])
        bow_indices=get_bag_of_words_indices(bow,tokenizer)
        bow_vector = build_bows_one_hot_vectors(bow_indices, tokenizer.vocab_size,tokenizer.name_or_path)
        output['bow_vector'] =bow_vector
    return output


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


def collate_fn(batch):
    if batch[0]["input_ids"][-1] == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0]["input_ids"][-1] == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False
    b = [
        [torch.tensor(item["input_ids"]), torch.tensor(item["output_ids"])]
        for item in batch
    ]
    input_ids, output_ids = list(zip(*b))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    
    output = {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "references": [item["reference"] for item in batch],
        "doc_id": [item["doc_id"] for item in batch],
        "entity": [item["entity"] for item in batch],
    }
    if "control_aspect_ids" in batch[0].keys():
        control_aspect_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item["control_aspect_ids"]) for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        output["control_aspect_ids"] = control_aspect_ids
    if "bow_vector" in batch[0].keys():
        bow_vector_batch=torch.stack([torch.tensor(item['bow_vector']) for item in batch])
        output['bow_vector_batch'] = bow_vector_batch

    return output


def get_dataloader_summ(
    dataset,
    tokenizer,
    batch_size,
    document_key,
    entity_key,
    summary_key,
    id_key,
    with_ent=False,
    shuffle=False,
    num_workers=0,
    max_length_input=1024,
    max_length_tgt=1024,
    return_ent_ids=False,
    use_pplm=False
):
    dataset = dataset.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "document_key": document_key,
            "entity_key": entity_key,
            "summary_key": summary_key,
            "id_key": id_key,
            "with_ent": with_ent,
            "max_length_input": max_length_input,
            "max_length_tgt": max_length_tgt,
            "return_ent_ids": return_ent_ids,
            "use_pplm":use_pplm
        },
        batched=False,
        remove_columns=dataset.column_names,
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dl


class Summarizer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if "ctrlsum" in args.model_name:
            if "pplm" in args.model_name:
                config = BartConfig.from_pretrained(
                    "hyunwoongko/ctrlsum-cnndm",
                    use_cache=True,
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
                self.model = PPLM(config,
                "hyunwoongko/ctrlsum-cnndm",
                cache_dir="%s/pretrained_models" % (args.data_folder),
                pplm_step_size=args.stepsize,gm_scale=args.gm_scale,model_type='ctrlsum'
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    "hyunwoongko/ctrlsum-cnndm",
                    use_cache=True,
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )

            self.tokenizer = BartTokenizer.from_pretrained(
                "hyunwoongko/ctrlsum-cnndm",
                use_cache=True,
                cache_dir="%s/pretrained_models" % (args.data_folder),
            )
        elif "bart" in args.model_name:
            if "pplm" in args.model_name:
                config = BartConfig.from_pretrained("facebook/bart-large-cnn",
                                                    cache_dir="%s/pretrained_models" % (args.data_folder))
                self.model = PPLM(config,
                    "facebook/bart-large-cnn",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                    pplm_step_size=args.stepsize,gm_scale=args.gm_scale,model_type='bart'
                )
            else:
                self.model = BartForConditionalGeneration.from_pretrained(
                    "facebook/bart-large-cnn",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
            self.tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large-cnn",
                cache_dir="%s/pretrained_models" % (args.data_folder),
            )
        elif "pegasus" in args.model_name:
            if 'pplm' in args.model_name:
                config = PegasusConfig.from_pretrained("google/pegasus-large",
                                                    cache_dir="%s/pretrained_models" % (args.data_folder))
                self.model = PPLM(config,
                    "google/pegasus-large",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                    pplm_step_size=args.stepsize,gm_scale=args.gm_scale,model_type='pegasus'
                )
            else:
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    "google/pegasus-large",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
            self.tokenizer = PegasusTokenizer.from_pretrained(
                "google/pegasus-large",
                cache_dir="%s/pretrained_models" % (args.data_folder),
            )
        elif "relattn" in args.model_name:
            if '-p' in args.model_name:
                self.config = PegasusRelConfig.from_pretrained(
                "google/pegasus-large",
                cache_dir="%s/pretrained_models" % (args.data_folder),
            )
                self.config.rel_attn_weight = args.rel_attn_weight
                self.config.rel_attn_type = args.rel_attn_type
                self.config.output_hidden_states = True
                self.config.smooth_method = args.smooth_method
                self.config.smooth_window = args.smooth_window
                self.config.smooth_gaussian_sigma = args.smooth_gaussian_sigma
                self.config.fixed_rel_attn_weight = not (args.learnable_rel_attn_weight)
                self.config.rel_attn_weight_perhead = args.rel_attn_weight_perhead
                self.config.rel_attn_weight_linear=args.rel_attn_weight_linear
                self.config.rel_attn_weight_with_ca_embed=args.rel_attn_weight_with_ca_embed
                self.model = PegasusRelForConditionalGeneration.from_pretrained(
                    "google/pegasus-large",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                    config=self.config,
                )
                self.tokenizer = PegasusTokenizer.from_pretrained(
                "google/pegasus-large",
                cache_dir="%s/pretrained_models" % (args.data_folder),
            )
            elif '-c' in args.model_name:
                self.config = BartRelConfig.from_pretrained(
                    "hyunwoongko/ctrlsum-cnndm",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
                self.config.rel_attn_weight = args.rel_attn_weight
                self.config.rel_attn_type = args.rel_attn_type
                self.config.output_hidden_states = True
                self.config.smooth_method = args.smooth_method
                self.config.smooth_window = args.smooth_window
                self.config.smooth_gaussian_sigma = args.smooth_gaussian_sigma
                self.config.fixed_rel_attn_weight = not (args.learnable_rel_attn_weight)
                self.config.rel_attn_weight_perhead = args.rel_attn_weight_perhead
                self.config.rel_attn_weight_linear=args.rel_attn_weight_linear
                self.config.rel_attn_weight_with_ca_embed=args.rel_attn_weight_with_ca_embed
                self.model = BartRelForConditionalGeneration.from_pretrained(
                    "hyunwoongko/ctrlsum-cnndm",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                    config=self.config,
                )
                self.tokenizer = BartTokenizer.from_pretrained(
                    "hyunwoongko/ctrlsum-cnndm",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
            else:
                self.config = BartRelConfig.from_pretrained(
                    "facebook/bart-large-cnn",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
                self.config.rel_attn_weight = args.rel_attn_weight
                self.config.rel_attn_type = args.rel_attn_type
                self.config.output_hidden_states = True
                self.config.smooth_method = args.smooth_method
                self.config.smooth_window = args.smooth_window
                self.config.smooth_gaussian_sigma = args.smooth_gaussian_sigma
                self.config.fixed_rel_attn_weight = not (args.learnable_rel_attn_weight)
                self.config.rel_attn_weight_perhead = args.rel_attn_weight_perhead
                self.config.rel_attn_weight_linear=args.rel_attn_weight_linear
                self.config.rel_attn_weight_with_ca_embed=args.rel_attn_weight_with_ca_embed
                self.model = BartRelForConditionalGeneration.from_pretrained(
                    "facebook/bart-large-cnn",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                    config=self.config,
                )
                self.tokenizer = BartTokenizer.from_pretrained(
                    "facebook/bart-large-cnn",
                    cache_dir="%s/pretrained_models" % (args.data_folder),
                )
            if args.rel_attn_weight_linear:
                for i in range(len(self.model.model.decoder.layers)):
                    self.model.model.decoder.layers[i].encoder_attn.rel_attn_weight[0].weight.data.zero_()
                    self.model.model.decoder.layers[i].encoder_attn.rel_attn_weight[0].bias.data.fill_(log(args.rel_attn_weight/(1-(args.rel_attn_weight))))
            if args.rel_attn_type == "trained":
                self.model.model.rel_k_proj.weight.data.fill_(0)
                self.model.model.rel_k_proj.weight.data.fill_diagonal_(1)
                self.model.model.rel_k_proj.bias.data.fill_(0)
                self.model.model.rel_q_proj.weight.data.fill_(0)
                self.model.model.rel_q_proj.weight.data.fill_diagonal_(1)
                self.model.model.rel_q_proj.bias.data.fill_(0)
        self.scorer = load_metric("rouge")
        if args.save_summary:
            summ_dir = "summary_%s__%s_%d_%d_beam=%d_lenPen=%.2f" % (
                self.args.dataset_name,
                self.args.model_name,
                self.args.max_length_input,
                self.args.max_length_tgt,
                self.args.beam_size,
                self.args.length_penalty,
            )
            summ_dir = (
                summ_dir
                + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
                if self.args.fewshot
                else summ_dir
            )
            self.summ_dir = os.path.join(args.output_folder, summ_dir)
            if not os.path.exists(self.summ_dir):
                os.makedirs(self.summ_dir)
        self.pad_token_id = self.tokenizer.pad_token_id

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(
        self,
        input_ids,
        output_ids,
        attention_mask=None,
        control_aspect_ids=None,
        control_aspect_mask=None,
    ):
        if not self.args.use_rel_attn:
            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=output_ids[:, :-1],
                attention_mask=attention_mask,
            )
        else:

            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=output_ids[:, :-1],
                rel_control_aspect_ids=control_aspect_ids,
                rel_control_aspect_mask=control_aspect_mask,
                attention_mask=attention_mask,
            )
        lm_logits = outputs.logits
        labels = output_ids[:, 1:].clone()

        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id
            )
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.tokenizer.pad_token_id,
            )
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input_ids = batch["input_ids"]
        output_ids = batch["output_ids"]
        if self.args.use_rel_attn:
            control_aspect_ids = batch["control_aspect_ids"]
            control_aspect_mask = torch.ones_like(control_aspect_ids).to(
                control_aspect_ids.device
            )
            control_aspect_mask[control_aspect_ids == self.pad_token_id] = 0
            attention_mask = torch.ones_like(input_ids).to(input_ids.device)
            attention_mask[input_ids == self.pad_token_id] = 0
            loss = self.shared_step(
                input_ids,
                output_ids,
                attention_mask,
                control_aspect_ids,
                control_aspect_mask,
            )
        else:
            loss = self.shared_step(input_ids, output_ids)
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size": input_ids.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        # Logging to TensorBoard by default
        self.log_dict(tensorboard_logs)
        return loss

    def generate(self, input_ids, control_aspect_ids=None, control_aspect_mask=None,bow_vector_batch=None,do_sample=True):
        if self.args.use_rel_attn:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.max_length_tgt
                if self.args.max_length_tgt
                else self.model.config.max_length,
                min_length=self.args.min_length_tgt
                if self.args.min_length_tgt
                else self.model.config.min_length,
                num_beams=self.args.beam_size
                if self.args.beam_size
                else self.model.config.num_beams,
                no_repeat_ngram_size=3
                if self.args.applyTriblck
                else self.model.config.no_repeat_ngram_size,
                length_penalty=self.args.length_penalty
                if self.args.length_penalty
                else self.model.config.no_repeat_ngram_size,
                rel_control_aspect_ids=control_aspect_ids,
                rel_control_aspect_mask=control_aspect_mask,
                do_sample=do_sample,
                temperature=1.5
            )
        elif self.args.perturb:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.max_length_tgt
                if self.args.max_length_tgt
                else self.model.config.max_length,
                min_length=self.args.min_length_tgt
                if self.args.min_length_tgt
                else self.model.config.min_length,
                num_beams=self.args.beam_size
                if self.args.beam_size
                else self.model.config.num_beams,
                no_repeat_ngram_size=3
                if self.args.applyTriblck
                else self.model.config.no_repeat_ngram_size,
                length_penalty=self.args.length_penalty
                if self.args.length_penalty
                else self.model.config.no_repeat_ngram_size,
                bow_vector_batch=bow_vector_batch,
                do_sample=do_sample,
                temperature=1.5
            )
        else:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.max_length_tgt
                if self.args.max_length_tgt
                else self.model.config.max_length,
                min_length=self.args.min_length_tgt
                if self.args.min_length_tgt
                else self.model.config.min_length,
                num_beams=self.args.beam_size
                if self.args.beam_size
                else self.model.config.num_beams,
                no_repeat_ngram_size=3
                if self.args.applyTriblck
                else self.model.config.no_repeat_ngram_size,
                length_penalty=self.args.length_penalty
                if self.args.length_penalty
                else self.model.config.no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=1.5
            )
        # pdb.set_trace()
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        return generated_str

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        # for p in self.model.parameters():
        #     p.requires_grad = False
        input_ids = batch["input_ids"]
        output_ids = batch["output_ids"]
        references = batch["references"]
        entity = batch["entity"]
        if self.args.use_rel_attn:
            control_aspect_ids = batch["control_aspect_ids"]
            control_aspect_mask = torch.ones_like(control_aspect_ids).to(
                control_aspect_ids.device
            )
            control_aspect_mask[control_aspect_ids == self.pad_token_id] = 0
            attention_mask = torch.ones_like(input_ids).to(input_ids.device)
            attention_mask[input_ids == self.pad_token_id] = 0
            loss = self.shared_step(
                input_ids,
                output_ids,
                attention_mask,
                control_aspect_ids,
                control_aspect_mask,
            )
        elif not self.args.perturb:
            loss = self.shared_step(input_ids, output_ids)
        else:
            loss = torch.zeros(input_ids.shape[0])
        if self.args.compute_rouge:
            doc_ids = batch["doc_id"]
            if self.args.use_rel_attn:
                generated_str = self.generate(
                    input_ids, control_aspect_ids, control_aspect_mask
                )
            elif self.args.perturb:
                # pdb.set_trace()
                generated_str=self.generate(
                    input_ids, bow_vector_batch=batch["bow_vector_batch"]
                )
            else:
                generated_str = self.generate(input_ids)
            generated_str, references = postprocess_text(generated_str, references)

            if self.args.save_summary:
                for i, gen_text in enumerate(generated_str):
                    with open(
                        os.path.join(self.summ_dir, "%s.txt" % (doc_ids[i])), "w",
                    ) as of:
                        of.write(gen_text)

            s = self.scorer.compute(
                predictions=generated_str,
                references=references,
                use_aggregator=False,
                use_stemmer=True,
            )
            sr_all = []
            pdb.set_trace()
            for i, gen_text in enumerate(generated_str):
                match = 0
                for ent in entity[i]:
                    if ent in gen_text:
                        match += 1
                if len(entity[i]) != 0:
                    sr = match / float(len(entity[i]))
                else:
                    sr = 0
                sr_all.append(sr)
            # pdb.set_trace()
            rouge_results = [
                (
                    s["rouge1"][i].recall,
                    s["rouge1"][i].precision,
                    s["rouge1"][i].fmeasure,
                    s["rouge2"][i].recall,
                    s["rouge2"][i].precision,
                    s["rouge2"][i].fmeasure,
                    s["rougeL"][i].recall,
                    s["rougeL"][i].precision,
                    s["rougeL"][i].fmeasure,
                    s["rougeLsum"][i].recall,
                    s["rougeLsum"][i].precision,
                    s["rougeLsum"][i].fmeasure,
                    sr_all[i],
                )
                for i in range(len(s["rouge1"]))
            ]
            return {"vloss": loss, "rouge_result": rouge_results, "doc_ids": doc_ids}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
        doc_ids_all = [i for b in outputs for i in b["doc_ids"]]
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        names.append("success_rate")
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc[len(rouge_results.index)] = avg
        doc_ids_all.append("avg")
        rouge_results = rouge_results.assign(doc_id=doc_ids_all)
        if self.args.save_rouge:
            csv_name = os.path.join(
                self.args.output_folder,
                output_file
                + "-%d.csv"
                % (torch.distributed.get_rank() if self.args.use_ddp else 0),
            )
            rouge_results.to_csv(csv_name)

        avgr = (avg[2] + avg[5] + avg[8]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, \
            Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        print("Success Rate: %f" % (metrics[12]))
        return names, metrics, avgr

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        self.model.train()
        # for p in self.model.parameters():
        #     p.requires_grad = True
        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        if self.args.compute_rouge:
            names, metrics, avgr = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.log_dict(logs, sync_dist=True if self.args.use_ddp else False)
            self.log("avgr", avgr, sync_dist=True if self.args.use_ddp else False)
            return {
                "avg_val_loss": vloss,
                "avgr": avgr,
                "r1f-avg": metrics[2],
                "r2f-avg": metrics[5],
                "rlf-avg": metrics[8],
                "rlsumf-avg": metrics[11],
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.log_dict(logs, sync_dist=True if self.args.use_ddp else False)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # tloss = torch.stack([x["vloss"] for x in outputs]).mean()
        tloss=0
        # self.log("tloss", tloss, sync_dist=True if self.args.use_ddp else False)

        output_file = "test_%s__%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.model_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgr = self.compute_rouge_all(outputs, output_file=output_file)
        metrics = [tloss, avgr] + metrics
        names = ["tloss", "avgr"] + names
        names = ["test_" + n for n in names]
        logs = dict(zip(*[names, metrics]))
        self.log_dict(logs, sync_dist=True if self.args.use_ddp else False)
        self.log("avgr_test", avgr, sync_dist=True if self.args.use_ddp else False)
        # self.log_dict(logs)
        return {
            "avg_test_loss": tloss,
            "avgr_test": avgr,
            "log": logs,
            "progress_bar": logs,
        }


def train(args):
    args.compute_rouge = True
    model = Summarizer(args)
    if args.learnable_parameters == "new_params_only":
        # only for relattn model
        for n, p in model.model.named_parameters():
            if ("rel_attn_weight" in n) or ("rel_k_proj" in n) or ("rel_q_proj" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif args.learnable_parameters == "cross_attn":
        # only for relattn model
        for n, p in model.model.named_parameters():
            if "encoder_attn" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif args.learnable_parameters == "cross_attn_k_q":
        # only for relattn model
        for n, p in model.model.named_parameters():
            if "encoder_attn.q_proj" in n or "encoder_attn.k_proj" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = True

    # pdb.set_trace()
    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.output_folder, "summ_checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgr:.4f}",
        save_top_k=args.saveTopK,
        monitor="avgr",
        mode="max",
        save_on_train_epoch_end=False,
    )

    # initialize logger
    logger_path = os.path.join(args.output_folder, "tb_logs")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger = TensorBoardLogger(logger_path, name="my_model")

    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps,
        accumulate_grad_batches=args.acc_batch,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,
        callbacks=checkpoint_callback,
        enable_checkpointing=True,
        precision=32,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        strategy="ddp" if args.use_ddp else None,
    )

    # load datasets
    train_dataset = None
    valid_dataset = None
    # placeholder
    
    if "newts" in args.dataset_name:
        dataset = load_from_disk("%s/data/NEWTS_dataset_with_general_summ" % (args.data_folder))[
            "train"
        ]
        document_key = "article"
        summary_key = "summary"
        id_key = "doc_id"
        entity_key = "topic_prefix"
        if "words" in args.dataset_name:
            # remove the last one, which is always empty
            dataset = dataset.map(
                function=lambda x: {
                    "topic_prefix": [
                        s.strip() for s in x["topic_words"].split(",")[:-1]
                    ]
                },
            )
        elif "phrases" in args.dataset_name:
            # no empty string in the list
            dataset = dataset.map(
                function=lambda x: {
                    "topic_prefix": [
                        s.strip() for s in x["topic_phrases"].split(",")
                    ]
                },
            )
        elif "sentences" in args.dataset_name:
            dataset = dataset.map(
                function=lambda x: {"topic_prefix": [x["topic_sentences"]]},
            )
    elif "entsum" in args.dataset_name:
        dataset = load_from_disk(
            "%s/data/entsum_final" % (args.data_folder)
        )["train"]
        dataset = dataset.map(
            function=lambda x: {
                "entity_text": [x["entity_text"]],
                "summary": " ".join(x["summary"]),
            },
            remove_columns=["entity_text", "summary"],
        )
        document_key = "article"
        entity_key = "entity_text"
        summary_key = "summary"
        id_key = "doc_id"
    
    all_doc_ids = list(set([d[id_key].split("-")[0] for d in dataset]))
    all_doc_ids.sort()
    np.random.shuffle(all_doc_ids)
    if args.fewshot:
        if args.num_valid_data == -1:
            args.num_valid_data = args.num_train_data
        i_doc_id = 0
        indices_train = []

        while len(indices_train) < args.num_train_data:
            doc_id = all_doc_ids[i_doc_id]
            for i, d in enumerate(dataset):
                if doc_id in d[id_key]:
                    indices_train.append(i)
            i_doc_id += 1
        indices_train = indices_train[: args.num_train_data]

        indices_valid = []
        while len(indices_valid) < args.num_valid_data:
            doc_id = all_doc_ids[i_doc_id]
            for i, d in enumerate(dataset):
                if doc_id in d[id_key]:
                    indices_valid.append(i)
            i_doc_id += 1
        indices_valid = indices_valid[: args.num_valid_data]
    else:
        # all_data_indices = list(range(len(dataset)))
        # np.random.shuffle(all_data_indices)
        # indices_train = all_data_indices[: int(0.9 * len(dataset))]
        # indices_valid = all_data_indices[int(0.9 * len(dataset)) :]
        indices_train = [
            i
            for i, d in enumerate(dataset)
            if d[id_key].split("-")[0] in all_doc_ids[: int(0.9 * len(all_doc_ids))]
        ]
        indices_valid = [
            i
            for i, d in enumerate(dataset)
            if d[id_key].split("-")[0] in all_doc_ids[int(0.9 * len(all_doc_ids)) :]
        ]
    train_dataset = dataset.select(indices_train)
    valid_dataset = dataset.select(indices_valid)
    print("train: ", indices_train[:10])
    print("valid: ", indices_valid[:10])

    tokenizer = model.tokenizer
    train_dataloader = get_dataloader_summ(
        train_dataset,
        tokenizer,
        args.batch_size,
        document_key,
        entity_key,
        summary_key,
        id_key,
        with_ent=args.with_ent,
        shuffle=True,
        num_workers=args.num_workers,
        max_length_input=args.max_length_input,
        max_length_tgt=args.max_length_tgt,
        return_ent_ids=args.use_rel_attn,
    )
    valid_dataloader = get_dataloader_summ(
        valid_dataset,
        tokenizer,
        args.batch_size,
        document_key,
        entity_key,
        summary_key,
        id_key,
        with_ent=args.with_ent,
        shuffle=False,
        num_workers=args.num_workers,
        max_length_input=args.max_length_input,
        max_length_tgt=args.max_length_tgt,
        return_ent_ids=args.use_rel_attn,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )
    if args.test_imediate:
        args.use_customize_ckpt = True
        args.ckpt_path = None
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    print("test started")
    args.save_summary = True
    args.save_rouge = True
    args.compute_rouge = True
    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.acc_batch,
        replace_sampler_ddp=False,
        # log_every_n_steps=5,
        enable_checkpointing=True,
        precision=32,
        accelerator=args.accelerator,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
    )

    if args.use_customize_ckpt:
        if args.ckpt_path is None:
            args.ckpt_path = os.path.join(args.output_folder, "summ_checkpoints")
        else:
            args.ckpt_path = os.path.join(args.ckpt_path, "summ_checkpoints")
        if os.path.isdir(args.ckpt_path):
            all_ckpt = list(Path(os.path.join(args.ckpt_path)).glob("*.ckpt"))
            # find the ckpt with the highest avgr score
            max_ind = np.argmax(
                [
                    float(re.search("(?<=avgr\=)[\d|.]+", all_ckpt[i].name[:-5])[0])
                    for i in range(len(all_ckpt))
                ]
            )
            args.ckpt_path = all_ckpt[max_ind]
        print("load from %s" % (args.ckpt_path))
        model = Summarizer.load_from_checkpoint(args.ckpt_path, args=args)
    else:
        model = Summarizer(args)
    # pdb.set_trace()
    # load dataset

    if "entsum" in args.dataset_name:
        if args.use_valid:
            # use validation set
            dataset = load_from_disk(
                "%s/data/entsum_final" % (args.data_folder)
            )["train"]
        else:
            # use test set
            dataset = load_from_disk(
                "%s/data/%s" % (args.data_folder, args.dataset_name)
            )["test"]
        
        dataset = dataset.map(
            function=lambda x: {
                "entity_text": [x["entity_text"]],
                "summary": " ".join(x["summary"]),
            },
            remove_columns=["entity_text", "summary"],
        )
        document_key = "article"
        entity_key = "entity_text"
        summary_key = "summary"
        id_key = "doc_id"

    elif "newts" in args.dataset_name:
        if args.use_valid:
            # use validation set
            dataset = load_from_disk("%s/data/NEWTS_dataset_with_general_summ" % (args.data_folder))[
                "train"
            ]
            all_data_indices = list(range(len(dataset)))
            np.random.shuffle(all_data_indices)
            dataset = dataset.select(all_data_indices[int(0.9 * len(dataset)) :])
        else:
            #use test set
            dataset = load_from_disk("%s/data/NEWTS_dataset_with_general_summ" % (args.data_folder))[
                "test"
            ]
        document_key = "article"
        summary_key = "summary"
        id_key = "doc_id"
        entity_key = "topic_prefix"
        if "words" in args.dataset_name:
            # remove the last one, which is always empty
            dataset = dataset.map(
                function=lambda x: {
                    "topic_prefix": [
                        s.strip() for s in x["topic_words"].split(",")[:-1]
                    ]
                },
            )
        elif "phrases" in args.dataset_name:
            # no empty string in the list
            dataset = dataset.map(
                function=lambda x: {
                    "topic_prefix": [s.strip() for s in x["topic_phrases"].split(",")]
                },
            )
        elif "sentences" in args.dataset_name:
            dataset = dataset.map(
                function=lambda x: {"topic_prefix": [x["topic_sentences"]]},
            )
    tokenizer = model.tokenizer
    test_dataloader = get_dataloader_summ(
        dataset,
        tokenizer,
        args.batch_size,
        document_key,
        entity_key,
        summary_key,
        id_key,
        with_ent=args.with_ent,
        shuffle=False,
        num_workers=args.num_workers,
        max_length_input=args.max_length_input,
        max_length_tgt=args.max_length_tgt,
        return_ent_ids=args.use_rel_attn,
        use_pplm=args.perturb
    )
    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########################
    # Gneral
    parser.add_argument("--gpus", default=0, type=int, help="number of gpus to use")
    parser.add_argument(
        "--accelerator", default="gpu", type=str, help="Type of accelerator"
    )
    parser.add_argument(
        "--mode", default="train", choices=["train", "test", "save_model"]
    )
    parser.add_argument(
        "--model_name", default="bart-cnndm",
    )
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )
    parser.add_argument(
        "--save_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        help="whether to save the generated summaries in validation steps",
    )

    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=3, type=int)

    parser.add_argument("--data_folder", type=str, default="./")
    parser.add_argument("--dataset_name", type=str, default="entsum")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=1024, type=int)
    parser.add_argument("--max_length_tgt", default=142, type=int)
    parser.add_argument("--min_length_tgt", default=56, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )

    ########################
    # For training
    parser.add_argument(
        "--limit_valid_batches", type=int, default=None,
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=150, help="Number of warmup steps"
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step"
    )
    parser.add_argument(
        "--total_steps", type=int, default=1500, help="Number of steps to train"
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )
    parser.add_argument(
        "--num_valid_data",
        type=int,
        default=-1,
        help="Number of validation data, -1 for full dataset and any positive number indicates how many data to use",
    )

    parser.add_argument(
        "--fix_lr", action="store_true", help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate", action="store_true", help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=4, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=2,
        help="length penalty of generated text",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )

    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )
    parser.add_argument(
        "--perturb", action="store_true", help="whether use pplm",
    )
    parser.add_argument(
        "--stepsize", type=float, default=0.0001, help="stepsize used for pplm",
    )
    parser.add_argument(
        "--gm_scale", type=float, default=0.95, help="gm_scale used for pplm",
    )
    parser.add_argument(
        "--with_ent", action="store_true", help="whether use entity",
    )
    parser.add_argument(
        "--use_ddp", action="store_true", help="whether use ddp as accelerator",
    )
    parser.add_argument(
        "--use_customize_ckpt", action="store_true", help="whether use customized_ckpt",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="check validation set within each epoch",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation set every n epochs",
    )
    parser.add_argument(
        "--use_rel_attn",
        action="store_true",
        help="whether to use relevance attention",
    )
    parser.add_argument(
        "--rel_attn_type",
        type=str,
        choices=["fixed", "trained"],
        help="whether to use fixed or trained relevance attention",
    )
    parser.add_argument(
        "--rel_attn_weight",
        type=float,
        default=0.1,
        help="weight of relevant attention",
    )
    parser.add_argument(
        "--use_valid", action="store_true", help="whether use validation set to test",
    )
    parser.add_argument(
        "--smooth_method",
        type=str,
        default=None,
        choices=["Gaussian"],
        help="smoothing method used for relattn",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=10,
        help="half window size of smoothing used for relattn",
    )
    parser.add_argument(
        "--smooth_gaussian_sigma",
        type=float,
        default=1,
        help="sigma used for gaussian smooth",
    )
    parser.add_argument(
        "--learnable_rel_attn_weight",
        action="store_true",
        help="whether use learnable rel weight",
    )
    parser.add_argument(
        "--learnable_parameters",
        type=str,
        choices=["all", "new_params_only", "cross_attn"],
        default="all",
        help="whether only train the new parameters",
    )
    parser.add_argument(
        "--rel_attn_weight_perhead",
        action="store_true",
        help="whether to have unique rel weights for each head on each layer",
    )
    parser.add_argument(
        "--rel_attn_weight_linear",
        action="store_true",
        help="whether to use linear layer to learn the rel weight for different data ",
    )
    parser.add_argument(
        "--rel_attn_weight_with_ca_embed",
        action="store_true",
        help="whether to include the cca embedding in the linear layer to learn the rel weight ",
    )
    args = parser.parse_args()  # Get pad token id
    ####################
    args.acc_batch = args.accum_data_per_step // args.batch_size
    model_folder = "%s_%s" % (args.dataset_name, args.model_name)
    if "relattn" in args.model_name:
        model_folder += "_%s" % (args.rel_attn_type)
        if args.rel_attn_weight_perhead:
            model_folder += "_perhead"
        else:
            model_folder += "_perlayer"
        if args.learnable_rel_attn_weight:
            model_folder += "_learned_weights"
        model_folder += "_%.2f" % (args.rel_attn_weight)
        if args.smooth_method == "Gaussian":
            model_folder += "_gaussian_%d_%f" % (
                args.smooth_window,
                args.smooth_gaussian_sigma,
            )
        model_folder += "_%s" % (args.learnable_parameters)

    model_folder += "_withent" if args.with_ent else ""
    model_folder += "_pplm_%f" % (args.stepsize) if args.perturb else ""
    model_folder +="_linear" if args.rel_attn_weight_linear else ""
    model_folder +="_with_ca" if args.rel_attn_weight_linear and args.rel_attn_weight_with_ca_embed else ""
    if args.fewshot:
        model_folder += "_fewshot_%d" % (
            args.num_train_data,
        )
    model_folder += "_seed=%d" % (args.rand_seed)
    args.output_folder = os.path.join(args.output_folder, model_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    seed_everything(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)

    print(args)
    with open(
        os.path.join(
            args.output_folder, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "save_model":
        if args.use_customize_ckpt:
            if args.ckpt_path is None:
                args.ckpt_path = os.path.join(args.output_folder, "summ_checkpoints")
            else:
                args.ckpt_path = os.path.join(args.ckpt_path, "summ_checkpoints")
            if os.path.isdir(args.ckpt_path):
                all_ckpt = list(Path(os.path.join(args.ckpt_path)).glob("*.ckpt"))
                # find the ckpt with the highest avgr score
                max_ind = np.argmax(
                    [
                        float(re.search("(?<=avgr\=)[\d|.]+", all_ckpt[i].name[:-5])[0])
                        for i in range(len(all_ckpt))
                    ]
                )
                args.single_ckpt_path = all_ckpt[max_ind]
            print("load from %s" % (args.single_ckpt_path))
            model = Summarizer.load_from_checkpoint(args.single_ckpt_path, args=args)
        else:
            model = Summarizer(args)
        print("save finetuned models")
        model.model.save_pretrained(
            os.path.join(args.ckpt_path, "..", "finetuned_model")
        )

