Much of the content (training script, utils, etc) of this directory is originally from: https://github.com/kimiyoung/transformer-xl/tree/master/pytorch

Some of the content has been modified by Matthew Kelleher

To run code you first need to download the desired dataset (see getdata.sh) and then update bash scripts (run_XXX.sh) to specify model and training parameters. 

README from original repo:
___ 

## Introduction

This directory contains our pytorch implementation of Transformer-XL. Note that our state-of-the-art results reported in the paper were obtained by training the model on a large-scale TPU cluster, and our pytorch codebase currently does not support distributed training. Here we provide two sets of hyperparameters and scripts:
- `*large.sh` are for the SoTA setting with large models which might not be directly runnable on a local GPU machine.
- `*base.sh` are for the base models which can be run on a few GPUs.

The pytorch implementation produces similar results to the TF codebase under the same settings in our preliminary experiments.


## Prerequisite

- Pytorch 0.4: `conda install pytorch torchvision -c pytorch`


## Data Prepration

`bash getdata.sh`

## Training and Evaluation

#### Replicate the "bpc = 1.06" result on `enwik8` with a 12-layer Transformer-XL

- Make sure the machine have **4 GPUs**, each with **at least 11G memory**

- Training

  `bash run_enwik8_base.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation

  `bash run_enwik8_base.sh eval --work_dir PATH_TO_WORK_DIR`



#### Replicate the "PPL = 24.03" result on `wikitext-103` with Transformer-XL

- Make sure the machine have **4 GPUs**, each with **at least 11G memory**

- Training

  `bash run_wt103_base.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation

  `bash run_wt103_base.sh eval --work_dir PATH_TO_WORK_DIR`



#### Other options:

- `--batch_chunk`: this option allows one to trade speed for memory. For `batch_chunk > 1`, the program will split each training batch into `batch_chunk` sub-batches and perform forward and backward on each sub-batch sequentially, with the gradient accumulated and divided by `batch_chunk`. Hence, the memory usage will propertionally lower while the computation time will inversely higher. 
- `--div_val`: when using adaptive softmax and embedding, the embedding dimension is divided by `div_val` from bin $i$ to bin $i+1$. This saves both GPU memory and the parameter budget.
- `--fp16` and `--dynamic-loss-scale`: Run in pseudo-fp16 mode (fp16 storage fp32 math) with dynamic loss scaling. 
  - Note: to explore the `--fp16` option, please make sure the `apex` package is installed (https://github.com/NVIDIA/apex/).
- To see performance without the recurrence mechanism, simply use `mem_len=0` in all your scripts.
- To see performance of a standard Transformer without relative positional encodings or recurrence mechanisms, use `attn_type=2` and `mem_len=0`.


#### Other datasets:

- `Text8` character-level language modeling: check out `run_text8_base.sh`
- `lm1b` word-level language modeling: check out `run_lm1b_base.sh`


___ 

## Transformer-XL
Created models with 8, 12, and 16 layers each with 8 attention heads. Models were trained with 10% droppout, learning rate 0.00025 and optimized with adam.

Models were trained on a NVIDIA RTX 2060. 


<img width="375px" src="src/transformer-xl/figures/enwik8_loss.png"/>

<img width="375px" src="src/transformer-xl/figures/wikitext2_loss.png"/>

<img width="375px" src="src/transformer-xl/figures/ptb_loss.png"/>

Note that for WikiText-2 due to memory constraints on the gpu we were only able to use a 15 layer model for our "large" model as opposed to the 16 layer models used for the other two datasets.
We also notice that while the large models do perform better the difference in performance is marginal. 
We observe very strange behavior with the 16 layer model on the Penn Tree Bank dataset. We retrained this model multiple times and adjusted model parameters all of which had little to no signifcant effect. We are currently unsure why this model performs like this and will need to investigate further.


##### Testset Results
|   | Perplexity | Avg ms/batch | 
|---|---|---|
|  EnWik8_8 | 4.705 | 234.76 |  
|  EnWik8_12 | 4.416 | 470.15  |  
|  EnWik8_16 | **4.393** | 472.20  |
|  WikiText-2_8 | 133.080 | 296.99 |
|  WikiText-2_12 | 126.021 | 387.55 |
|  WikiText-2_15 | **123.108** | 495.67 |
|  PTB_8 | 96.605 | 248.05 |
|  PTB_12 | **91.249** | 360.85 |
|  PTB_16 | 651.929 | 490.60 |
