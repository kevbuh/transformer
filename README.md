# Recreating a Modern Transformer in PyTorch

"What I cannot create, I do not understand" - Richard Feynman

### Main model Based of the Paper by Google's Brain Research Team, "An Image Is Worth 16 x 16 Words".

### Go to vit.ipynb for Visual transformer works

First model is based of the paper, "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf.
Lots of help from https://peterbloem.nl/blog/transformers and https://jalammar.github.io/illustrated-transformer/.

Notes and code are listed below and in model.ipynb

# Notes on general Transformers

### How Self Attention Works

This is the big breakthrough in how Transformers became a state of the art architecture, and are fundamental to all Transformers.
This self attention is called 'sequence to sequence', because a sequence of vectors go in are then are pushed out.
The output vectors are created by a self attention operation that takes a weighted average of all the input vectors that loop through the entire sequence. Unlike in normal neural networks, the weights in the equation are not parameters, but come from the dot product of the input vector and the current sequence vector. Self attention also has a softmax() function to map the values from 0 - 1, which is the range neural networks normal work with. Self attention is also in charge of propagating information between vectors.
Self attention works similar to how recommendation systems work. The model learns how correlated words in the input sequence are based off of the dot product of input vectors and the known weights. The output vectors are weighted sums from the entire input sequence whose weights come from the dot products.

<img src="https://render.githubusercontent.com/render/math?math={\color{black} \displaystyle\sum_{d=0}^{d_{max}}}">

Below is a simple Transformer self attention in PyTorch.
The input is a sequence of t vectors with k dimensions and in code is given as a matrix X.
The batch size is dimension b, so this leaves us with an input tensor of (b,t,k).

```
import torch
import torch.nn.functional as F

# - bmm = batched matrix multiplication.
#   Applies matrix multiplication over batches of matrices.
raw_weights = torch.bmm(x, x.transpose(1, 2))

# turns all raw weights into positive values that some to one
# uses row-wise softmax.
weights = F.softmax(raw_weights, dim=2)

# Computes output sequence
# multiply the weight matrix by initial matrix X
# Results in batch of output matrices Y of size (b, t, k)
# Each row in Y is a weighted sum of the rows in initial matrix X.
y = torch.bmm(weights, x)
```

Above was basic self attention, modern Transformers have additional parts to attention.

One of which is the use of queries (Q), keys (K), and values (V) as mentioned in the original paper. Each of the input vectors is compared to every other to create weights for its own output y_i, the output of the j-th vector y_j, and is included in the weighted sum that creates the output vector.
\*\*\* need more info on QKV

Softmax functions can potentially kill training if input values are too big, so the paper includes a scaling to prevent too large of input values. They divide by sqrt(k) because it is "dividing out the amount by which the increase in dimension increases the length of the average vectors".

The paper also mentions multi-head attention. They include attention heads, which are weight matrices for the QKV. In result this creates copies of the self-attention mechanism, applied parallel and with their own KVQ transformations.

```
class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)
```

# Overview of Transformer Blocks

From input to output, the big picture view goes from self attention layer, to layer normalization, to a feed forward layer with multi layer perceptrons, and finally through another layer normalization. Residual connections are also strung before the normalization.

```
class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = SelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)

    fedforward = self.ff(x)
    return self.norm2(fedforward + x)
```
