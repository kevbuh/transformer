# Recreating a Modern Transformer in PyTorch

"What I cannot create, I do not understand" - Richard Feynman

Based of the paper, "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf.
Lots of help from https://peterbloem.nl/blog/transformers and https://jalammar.github.io/illustrated-transformer/.

Notes and code are listed below and in model.ipynb

## Self Attention

This is the big breakthrough in how Transformers became a state of the art architecture, and are fundamental to all Transformers.
This self attention is called 'sequence to sequence', because a sequence of vectors go in are then are pushed out.
The output vectors are created by a self attention operation that takes a weighted average of all the input vectors that loop through the entire sequence. Unlike in normal neural networks, the weights in the equation are not parameters, but come from the dot product of the input vector and the current sequence vector. Self attention also has a softmax() function to map the values from 0 - 1, which is the range neural networks normal work with. Self attention is also in charge of propagating information between vectors.
Self attention works similar to how recommendation systems work. The model learns how correlated words in the input sequence are based off of the dot product of input vectors and the known weights. The output vectors are weighted sums from the entire input sequence whose weights come from the dot products.

<img src="https://render.githubusercontent.com/render/math?math={\color{black} \displaystyle\sum_{d=0}^{d_{max}}}">

```
function test() {
  console.log("notice the blank line before this function?");
}
```
