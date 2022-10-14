# Recreating a Modern Transformer in PyTorch

Based of the paper, "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf.
Lots of help from https://peterbloem.nl/blog/transformers and https://jalammar.github.io/illustrated-transformer/.

Notes and code are listed below and in model.ipynb

## Self Attention

This is the big breakthrough in how Transformers became a state of the art architecture, and are fundamental to all Transformers.
This self attention is called 'sequence to sequence', because a sequence of vectors go in are then are pushed out.
The output vectors are created by a self attention operation that takes a weighted average of all the input vectors.
<img src="https://render.githubusercontent.com/render/math?math={\color{black} \displaystyle\sum_{d=0}^{d_{max}}}">
