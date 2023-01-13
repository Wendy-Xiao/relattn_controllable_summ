# RelAttn
The official code for 'Attend to the Right Context: A Plug-and-Play Module for Content-Controllable Summarization'

## Environment
The configure file of the virtual environment for our experiments is in `pplm_env_latest.yaml`.
You can create an identical environment using 
```
conda create -n relattn_env -f pplm_env_latest.yaml
```

## Datasets
We use NEWTS and ENTSum for our experiments. The original datasets can be found here: [NEWTS](https://github.com/ali-bahrainian/NEWTS) and [ENTSum](https://github.com/bloomberg/entsum).

Our pre-processed data can be found [here](https://drive.google.com/drive/folders/17_WHJH7qjPsobjyKhzSMJeYn1YkD-NDD?usp=share_link).

## Models
The implementation of the PPLM model is in `./PPLM.py`.

The implementation of BART+RelAttn and PEGASUS+Relattn is in `./RelAttnModel.py` and `./pegasusRelAttnModel.py`, respectively.

## Inference Process
The three backbone models can be selected by specifying in `--model_name`,
* BART: include `bart` in the model_name
* PEGASUS: include `pegasus` in the model_name
* CTRLSum: include `ctrlsum` in the model_name 
### PPLM
To use PPLM, 
* include `pplm` in the model_name, e.g. use bart as the backbone model with pplm, just set `--model_name` as `bart-pplm`
* add `--perturb`
* specifying `--stepsize` and `--gm_scale`

An example command for running PPLM with bart as backbone model on the `newts` dataset.
```
python ./run_summarizaion_pl.py --gpus 1 
                                --mode test 
                                --output_folder ./output/
                                --data_folder ./data/
                                --dataset_name newts-words 
                                --model_name bart-pplm
                                --perturb
                                --save_rouge
                                --stepsize 1e-5
                                --gm_scale 0.65 
                                --max_length_tgt 142 
                                --min_length_tgt 56 
                                --applyTriblck
```

### BART/CTRLSum/PEGASUS - CA+Doc
To use CA+Doc as the input, simply add `--with_ent` when running the code.

```
python ./run_summarization_pl.py --gpus 1 
                                --mode test 
                                --output_folder ./output/
                                --data_folder ./data/
                                --dataset_name newts-words 
                                --model_name bart 
                                --with_ent
                                --max_length_tgt 142 
                                --min_length_tgt 56 
                                --applyTriblck 
```
### RelAttn
To use RelAttn, 
* Specifying `--model_name` as one of `relattn-b`,`relattn-p` and `relattn-c`, indicating BART, PEGASUS and CTRLSum as backbone models respectively.
* Add `--use_rel_attn`, `--rel_attn_type fixed` and desired `--rel_attn_weight`
* Add `--smooth_method Gaussian`

An example command to apply RelAttn on the BART model 
```
python ./run_summarization_pl.py --gpus 1 
                                --mode test 
                                --output_folder ./output/
                                --data_folder ./data/
                                --dataset_name newts-words
                                --model_name relattn-b 
                                --use_rel_attn 
                                --rel_attn_type fixed 
                                --rel_attn_weight 0.12
                                --smooth_method Gaussian
                                --max_length_tgt 142 
                                --min_length_tgt 56 
                                --applyTriblck
```
### OS algorithm
In our implementations, we first generate all the summaries with different rel_weight ranges [0.01,0.30], and select the best one using the online selection algorithm in `./Online_Selection.py`

## Training Process
To train the models (except for PPLM models) in few-shot settings,
* Specifying `--mode train` and add `--fewshot`
* Sepcifying `----num_train_data`, `--total_steps`, `--warmup_steps`,`--batch_size` and `--lr` 

An example command to train `relattn-b` in few-shot settings
```
python ./run_summarization_pl.py --gpus 1 
                                --mode train 
                                --output_folder ./output/
                                --data_folder ./data/
                                --dataset_name newts-words
                                --model_name relattn-b 
                                --save_rouge 
                                --use_rel_attn 
                                --rel_attn_type trained 
                                --rel_attn_weight 0.07
                                --learnable_rel_attn_weight
                                --rel_attn_weight_linear
                                --rel_attn_weight_with_ca_embed
                                --max_length_tgt 142 
                                --min_length_tgt 56
                                --total_steps 100 
                                --warmup_steps 10 
                                --batch_size 5
                                --lr 3e-5
                                --fewshot 
                                --test_imediate
                                --applyTriblck
                                --num_train_data 10
                                --check_val_every_n_epoch 1 
                                --accum_data_per_step 10 
                                --rand_seed 1234
```
## Citation
```
@misc{https://doi.org/10.48550/arxiv.2212.10819,
  doi = {10.48550/ARXIV.2212.10819},
  url = {https://arxiv.org/abs/2212.10819},
  author = {Xiao, Wen and Miculicich, Lesly and Liu, Yang and He, Pengcheng and Carenini, Giuseppe},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Attend to the Right Context: A Plug-and-Play Module for Content-Controllable Summarization},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

