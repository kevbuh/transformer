# Coding a ML Transformer in PyTorch

Based of the paper, "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

Why?
Transformers can reach a new state of the art in
translation quality. They are also being used in the most advanced computer vision systems.

Goals:

- Train model to show X amount accuracy
- Build entire thing in PyTorch

## Encoder and Decoder

Most competitive neural sequence transduction models have an encoder-decoder structure.
"Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence
of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output
sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next."

### Encoder

- The encoder is composed of a stack of N = 6 identical layers.
- Each layer has two
  sub-layers.
  - The first is a multi-head self-attention mechanism, and the
  - second is a simple, positionwise fully connected feed-forward network.
- We employ a residual connection [11] around each of
  the two sub-layers, followed by layer normalization [1].
  - That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
    itself.
- To facilitate these residual connections, all sub-layers in the model, as well as the embedding
  layers, produce outputs of dimension d_model = 512.

### Decoder

- The decoder is also composed of a stack of N = 6 identical layers.
- In addition to the two
  sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
  attention over the output of the encoder stack.
- Similar to the encoder, we employ residual connections
  around each of the sub-layers, followed by layer normalization.
- We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
- This masking, combined with fact that the output embeddings are offset by one position, ensures that the
  predictions for position i can depend only on the known outputs at positions less than i.

## Attention-

- Mapping a query and a set of key-value pairs to an output,
  where the query, keys, values, and output are all vectors
- The output is computed as a weighted sum
  of the values, where the weight assigned to each value is computed by a compatibility function of the
  query with the corresponding key.

## Embedding Inputs and Softmax

- They use learned embeddings to convert the input
  tokens and output tokens to vectors of dimension d_model.

## 2. Positional Encodings

## 3. Creating Masks

## 4. Multi-Head Attention Layer

Relying entirely on an attention mechanism to draw global dependencies between input and output.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
of a single sequence in order to compute a representation of the sequences.

## 5. Feed Forward Layers

- Each of the layers in our encoder and decoder contains a fully
  connected feed-forward network
- Two linear transformations with a ReLU activation in between
