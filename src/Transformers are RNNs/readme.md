# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
The model is based on [this paper](https://arxiv.org/pdf/2006.16236.pdf).

We borrow the code for model from [this repo](https://github.com/idiap/fast-transformers) and some preprocessing code from [pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

## How to run the code
For WikiText2 and PennTreebank

`python train.py` 

For Enwik8

`python train_E.py`

The testing scripts are also in the training scripts, just set TEST=True in the script. Also, the hyperparameters and training settings are set by the `config` dictionary in the script. 

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
From the figure, we can see that batch size=16 yield the best results. For WikiText2 and PennTreebank, there is not a big difference between batch size of 16 and 64; for Enwik8, there is little difference between batch size of 4 and 16. When batch size is too small, the gradient step each update is too random and might not lead the parameters to a good local minimum. When the batch size is too large, one possible reason for its inferior performance is the number of examples becomes smaller if we train for equal number of epochs.

![image](https://user-images.githubusercontent.com/35536646/135794907-ea726dfa-5a91-405f-a404-37b6367d7e03.png)


### Varying Learning Rate
We can see that the learning rate of 0.00005 is the best across the two datasets (WikiText2 and PennTreebank). Due to the training time, we just use the best learning rate (5e-5) for enwik8.  

**WikiText2**

![Webp net-resizeimage (4)](https://user-images.githubusercontent.com/35536646/135793976-55291923-53ae-4f9a-9d4e-66b658762910.png)
![Webp net-resizeimage (5)](https://user-images.githubusercontent.com/35536646/135912990-f90bc59b-db35-4acb-9b12-7ca13e193497.png)


**PennTreebank**

![Webp net-resizeimage](https://user-images.githubusercontent.com/35536646/135790231-8f61c1f9-384b-45b1-beb9-b440056d1413.png)
![Webp net-resizeimage (6)](https://user-images.githubusercontent.com/35536646/135913011-432fa64c-351e-4491-bec6-4bfef112441e.png)


## Results on Number of Parameters
We try to vary the number of parameters, which affects the capacity of the model. We experiment with models of 2, 5, 8, 11 layers and observe how these models perform on the end tasks. 


### Varying Number of Layers
As the model becomes deeper, the perplexity tends to decrease. This indicates that larger model capacity can enhace model performance. 

![image](https://user-images.githubusercontent.com/35536646/135784476-02294812-3857-429b-9986-74206fbf6f7b.png)



## Comparison of Attention Type
As can be seen in the figure below, different attention types seem not to affect the result too much (see the scale). Note here we use the best-performing hyperparameters on linear attention models, but we can assuem the two models would performa similarly when carefully tuned. On language modeling tasks, transformer with linear attention does not sacrifice performance.

![image](https://user-images.githubusercontent.com/35536646/135793845-2bd830ad-5a77-47a1-ac2a-75ab60f6b888.png)


## Comparison of Sequence Length
For WikiText2 and PennTreebank dataset, shorter sequence length seems to bring better results. One possible reason is that for longer texts, the transformer might need to encode longer dependency, which can be more complicated for a medium-sized model (only 8 layers). For enwik8, longer sequences give better result. We think the reason is that this is a character-level language modeling dataset, which might need longer dependency to determine the next character. (e.g. 250 characters might only be about 30 words)

![image](https://user-images.githubusercontent.com/35536646/135793803-6ee1fec1-84d7-4d9a-87ba-95ed20da78a5.png)


## Run Time Comparison
Clearly Linear attention has faster run time. However, the difference is not that obvious due to the small size of the datasets and the length of sequence is not too long. (We use length of 250 in these experiments.)

|   | Linear | Full | 
|---|---|---|
|  WikiText2 | 3.58 | 3.84 |  
|  PennTreebank | 0.98 | 1.09 |  
|  EnWik8 | 26.89 | 27.37  | 


