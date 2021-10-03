# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
The model is based on [this paper](https://arxiv.org/pdf/2006.16236.pdf).

We borrow the code for model from [this repo](https://github.com/idiap/fast-transformers) and some preprocessing code from [pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

## Experiments

- Hyperparameters
  - Different Batch Size
  - Different Learning Rates
- Numbers of parameters
  - Different Layers
  - Different Attention Heads
- N^2 v.s. N  
- Varying sequence length
