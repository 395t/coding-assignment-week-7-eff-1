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
From the figure, we can see that batch size=16 yield the best results. There is not a big difference between batch size of 16 and 64 though. When batch size is too small, the gradient step each update is too random and might not lead the parameters to a good local minimum. When the batch size is too large, one possible reason for its inferior performance is the number of examples becomes smaller if we train for equal number of epochs.


### Varying Learning Rate
We can see that the learning rate of 0.00005 is the best across the two datasets (WikiText2 and PennTreebank). Due to the training time, we just use the best learning rate (5e-5) for enwik8.  

![train_P](https://user-images.githubusercontent.com/35536646/135790003-89099e91-ba6a-44ad-8152-37d0e6f7f6b0.png)

![valid_P](https://user-images.githubusercontent.com/35536646/135790089-6d797016-86fc-4193-a256-7bd91c5d151a.png)

## Results on Number of Parameters
We try to vary the number of parameters, which affects the capacity of the model. We experiment with models of 2, 5, 8, 11 layers and observe how these models perform on the end tasks. 


### Varying Number of Layers
As the model becomes deeper, the perplexity tends to decrease. This indicates that larger model capacity can enhace model performance. 

![image](https://user-images.githubusercontent.com/35536646/135784476-02294812-3857-429b-9986-74206fbf6f7b.png)



## Comparison of Attention Type
As can be seen in the figure below, different attention types seem not to affect the result too much (see the scale). Note here we use the best-performing hyperparameters on linear attention models, but we can assuem the two models would performa similarly when carefully tuned. On language modeling tasks, transformer with linear attention does not sacrifice performance.



## Comparison of Sequence Length

