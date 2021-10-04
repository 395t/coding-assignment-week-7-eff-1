# Reformer

## Model Architecture & Hyperparams
* 6-layer  
* 256-hidden  
* 2-heads  
* 3M parameters   
* Adafactor optimizer
* Learning Rate: 0.0002    
* Weight Decay: 0.01
* Batch Size: 16
* Epochs: 30 

## Task and Datasets
* Causal language Modeling  
* wikitext-2 | penn treebank | enwik8  
* Trained using Google Colab Pro  

## Summary  
Reformer is a highly memory efficient transformer model. In these experiments, Axial Position Encodings were not used, as the Huggingface implementation of fine tuning a pretrained model did not offer straightforward compatibility with the the axial encodings activated (requiring sequence lengths of 500k tokens). Therefore, the model was tuned and tested on sequence lengths of 256, 2048, and 16k tokens, as longer sequences were not feasible due to memory limitations. The experiments demonstrated that an increase in sequence length leads to faster training times at the cost of higher perplexity upon evaulation. 

A pretrained Reformer model was used, specifically a variant trained by Google AI on an English translation of the novel Crime and Punishment by Fyodor Dostoyevsky. This model uses subword level tokenization, and the tokenizer trained on the same text was used for all experiments. While the model was tuned on the training sets, the tokenizer was not tuned on the datasets, thus using a fixed vocabulary across all experiments. This approach was selected to hold as many variables constant while experimenting with different sequence lengths. 

## Results  
NOTE: Sequence Length of 512 instead of 256 used for enwik8 due to training time. (> 24hr)
### Evaluation Perplexity
| Seq Len  | wt2 | ptb | ew8 |
|---|---|---|---|
|  256 | 14.49 | **13.29**  |  0.00 |
|  2048 | 26.18 |  **24.54** |  32.99 |
|  16384 | 0.00  | 0.00  |  0.00 |

### Training Metrics per Sequence Length
#### Training Runtime
| Seq Len  | wt2 | ptb | ew8 |
|---|---|---|---|
|  256 | 1:11:32.24 | 0:29:33.46  |  0.00 |
|  2048 | 0:33:10.46 |  0:13:24.65 |  4:07:58.78 |
|  16384 | 0.00  | 0.00  |  0.00 |

#### Training Speed
| Seq Len  | Steps/sec |
|---|---|
|  256 | ~ 10.1 |
|  2048 | ~ 2.7 |
|  16384 | 0.00  |

#### Sample Speed
| Seq Len  | Samples/sec |
|---|---|
|  256 | ~ 161 |
|  2048 | ~ 43 |
|  16384 | 0.00  |

## Reference
Pretrained model, tokenizer, and examples sourced from Huggingface.
https://huggingface.co/blog/reformer  
https://huggingface.co/transformers/training.html    
https://huggingface.co/transformers/perplexity.html
