# Efficient Transformers

## Code Structure

## Task and Datasets

## Transformer-XL

<img width="375px" src="src/transformer-xl/figures/enwik8_loss.png"/>

<img width="375px" src="src/transformer-xl/figures/wikitext2_loss.png"/>

<img width="375px" src="src/transformer-xl/figures/ptb_loss.png"/>

## Sparse Transformer

## Compressive Transformer

Our base model had 8 layers with 8 attention heads. The memory and compressed memory sizes match the sequence length, which was 512. We used the compressed memory ratio recommended in the paper, which was 4.

We optimized the model using Adam with a learning rate of 1e-4.

### Sequence Length

Somewhat puzzlingly, using a sequence length of 2048 took four times as a long to train as our base model of 512, yet smaller sequence lengths trained in approximately the same amount of time as base.

![enwik8](https://user-images.githubusercontent.com/34489261/135774640-450cdbe7-95c9-4928-8a8c-463d648c6231.png)
![WikiText-2](https://user-images.githubusercontent.com/34489261/135774642-8c584f06-dbc9-40bb-88a7-c0583526a39b.png)
![PennTreebank](https://user-images.githubusercontent.com/34489261/135774673-354e1437-b0d1-41e9-8848-b889b3274521.png)

Overall, sequence length did not affect performance significantly. We see that only for enwik8 did having a long sequence length improve performance. For the other two datasets, sequence lengths of 128 to 512 performed well comparably. The reason a longer sequence length performs worse could possibly be explained by the fact that there are little dependencies between words that are so far apart. Only for character level modeling are there any meaningful dependencies. On the other hand, shorter sequence lengths hide information that could be helpful in processing the current tokens.

### Model Depth

Training time correlated linearly with model depth.

![enwik8](https://user-images.githubusercontent.com/34489261/135784527-e823d583-78be-4afb-bcb6-ab962c04346a.png)
![WikiText-2](https://user-images.githubusercontent.com/34489261/135784531-927f048e-7411-47d8-8cb6-e9da832e0bda.png)
![PennTreebank](https://user-images.githubusercontent.com/34489261/135784534-9e078f52-52e5-4a45-9a17-2694b794d3af.png)

We see trends that are similar to the ones for sequence length. Performance was again not affected very much.

## Reformer

## Transformers are RNNs


The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.

## Reference

Any code that you borrow or other reference should be properly cited.
