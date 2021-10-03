# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
The model is based on [this paper](https://arxiv.org/pdf/2006.16236.pdf).

We borrow the code for model from [this repo](https://github.com/idiap/fast-transformers) and some preprocessing code from [pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

## Experiments
### 
- Hyperparameters
  - Different batch size
  - Different learning rates
- Number of parameters
  - Different numbers of layers
  - Different numbers of attention heads
- Linear attention v.s. full attention 
- Varying sequence length

## Results on Hyperparameters
### Varying Batch Size

### Varying Learning Rate
We can see that the learning rate of 0.0005 is the best across different datasets.

## Results on Number of Parameters
### Varying Number of Layers
As the model becomes deeper, the perplexity tends to decrease. This indicates that larger model capacity can enhace model performance.

### Varying Number of Attention Heads

## Comparison of Attention Type
As can be seen in the figure below, different attention types seem not to affect the result too much. On language modeling tasks, transformer with linear attention does not sacrifice performance.


## Comparison of Sequence Length
