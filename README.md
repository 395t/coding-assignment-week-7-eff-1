# Efficient Transformers

## Code Structure

## Task and Datasets

## Transformer-XL

## Sparse Transformer

## Compressive Transformer

I used a 8-layer model with 8 attention heads.

I optimized my model using the Adam Optimizer with a learning rate of 1e-4.

![enwik8](https://user-images.githubusercontent.com/34489261/135774640-450cdbe7-95c9-4928-8a8c-463d648c6231.png)
![WikiText-2](https://user-images.githubusercontent.com/34489261/135774642-8c584f06-dbc9-40bb-88a7-c0583526a39b.png)
![PennTreebank](https://user-images.githubusercontent.com/34489261/135774673-354e1437-b0d1-41e9-8848-b889b3274521.png)

Overall, sequence length did not affect performance drastically. We see that only for enwik8 did having a long sequence length improve performance. For the other two datasets, sequence lengths of 128 to 512 performed well comparably. The reason a longer sequence length performs worse could possibly be explained by the fact that there are little dependencies between words that are so far apart. Only for character level modeling are there any meaningful dependencies. On the other hand, shorter sequence lengths hide information that could be helpful in processing the current tokens.

## Reformer

## Transformers are RNNs


The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.

## Reference

Any code that you borrow or other reference should be properly cited.
