#Lesson 1:Attention Mechanism
Imagine preparing for an exam.
Your notebook contains
formulas
jokes
examples
definitions
diagrams
When solving a derivative problem,
you don't read every page equally.
You immediately focus on
Formula page
That is attention.

Query, Key and Value

This is the core idea of modern attention.

Every input vector is transformed into three vectors:

Query (Q)


Key (K)


Value (V)

Think of them like a search engine.

Suppose Google.

Query

"Best Computer Vision books"

Keys

Book1
Book2
Book3
Book4

Google compares your query against every key.

The best matches receive higher scores.

Then Google returns their information.

Returned information

Value vectors
Hence
Query-asks
Keys-answer
Values-contain information

#Visual Flow
Input Features
↓
Create Q
Create K
Create V
↓
Q × K
↓
Similarity Scores
↓
Softmax
↓
Attention Weights
↓
Multiply by V
↓
Output Features
This is the complete attention pipeline.

#Self-Attention
This is the breakthrough behind Transformers.
Instead of comparing one sentence with another,
every token compares itself with every other token.
Example
Dog chased cat because it ran.
When processing
it
the model compares against
Dog
chased
cat
because
ran
to determine the most relevant context.

#Why is Self-Attention Better than CNNs or RNNs?

| Model          | Strength                            | Weakness                                      |
| -------------- | ----------------------------------- | --------------------------------------------- |
| RNN            | Sequence modeling                   | Slow, hard to capture long-range dependencies |
| CNN            | Local feature extraction            | Limited receptive field without many layers   |
| Self-Attention | Global relationships from the start | Computationally expensive for long sequences  |

For images, this means a transformer can directly relate the top-left corner to the bottom-right corner in one layer, while a CNN needs many layers before those regions influence each other.

#In image classification, why could allowing every image patch to attend to every other patch improve recognition of objects whose parts are far apart?
->
Allowing every image patch to attend to every other patch—the core mechanism of global self-attention in Vision Transformers (ViTs)—improves recognition of spatially distributed object parts by resolving the receptive field bottleneck inherent to traditional Convolutional Neural Networks (CNNs).

#Lesson 2:Transformer Basics:
A Transformer is not a single attention layer.

It is a stack of blocks, where each block contains:

Multi-Head Self-Attention
Add & Layer Normalization
Feed-Forward Network (MLP)
Add & Layer Normalization

This block is repeated many times.

Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Add + LayerNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
Add + LayerNorm
  │
  ▼
Repeat...

Multi-Head Attention

Instead of using one attention mechanism,

we use several in parallel.

Head 1 → local texture
Head 2 → global shape
Head 3 → color relationships
Head 4 → object boundaries

Each head can learn a different type of relationship.

The outputs are concatenated and projected back into a single representation.

Residual Connections

Instead of learning a complete new representation,

the block learns a refinement.

Output = Input + Block(Input)

Benefits:
Easier optimization.
Better gradient flow.
Enables very deep models.
Layer Normalization
Different features can have very different scales.
Layer normalization stabilizes activations within each sample, making training faster and more stable.
Feed-Forward Network (FFN)
After attention mixes information across tokens,
each token independently passes through a small MLP.
Typical structure:
Linear
↓
GELU / ReLU
↓
Linear
Attention handles interactions between tokens; the FFN increases the expressive power of each token's representation.

Complete Transformer Encoder Block
Input
 │
 ▼
Multi-Head Self-Attention
 │
 ▼
Add (Residual)
 │
 ▼
LayerNorm
 │
 ▼
Feed-Forward Network
 │
 ▼
Add (Residual)
 │
 ▼
LayerNorm
 │
 ▼
Output

This encoder block is what Vision Transformers repeatedly stack.

#Positional encodings are necessary because the self-attention mechanism in Transformers is mathematically permutation invariant
Without positional encodings, the set of input vectors {x_1, x_2, x_3} produces the exact same set of output attention representations regardless of how you shuffle their order. Since language and spatial image patches rely heavily on order and structure (e.g., *"Dog bites man"* vs. *"Man bites dog"*), positional encodings inject deterministic or learned positional vectors directly into token embeddings so the model can distinguish word or patch sequence order.
Difference Between Single-Head and Multi-Head Attention
->Single-Head Attention:** Computes a single set of Query, Key, and Value ($Q, K, V$) projections. It generates a single matrix of attention weights over the entire hidden dimension, forcing the network to average out different types of contextual relationships into one single distribution.
->Multi-Head Attention:** Splits the model's hidden dimension model into h distinct head sub-spaces d_k = d_model / h. It independently projects $Q, K, V$ for each head in parallel, allowing the model to jointly attend to information from **multiple representation subspaces** simultaneously (e.g., one head can focus on syntactic agreements while another focuses on semantic relationships or distant spatial visual parts).

---

#Why Residual Connections Are Important for Deep Transformers:

1. Mitigating Vanishing Gradients:** Transformer blocks use deep stacks of multi-head attention and multi-layer perceptrons (MLPs). Residual connections ($x + f(x)$) create unimpeded, linear highways for gradients to flow directly back to initial layers during backpropagation.
2. Preserving Identity Signals:** In attention layers, weighted sum operations continuously re-mix representations. Skip connections guarantee that a token’s original representation is preserved and added back, preventing deep networks from losing fine-grained low-level information through degradation.
#Component Roles in Information Processing
->Mixes information across tokens:** The **Multi-Head Self-Attention** layer. It aggregates feature vectors across different spatial/sequence positions based on their dynamic affinity scores.
->Processes each token independently:** The **Feed-Forward Network (FFN / MLP)** layer. It applies identical, position-wise non-linear transformations (typically two linear layers with an activation like GELU/ReLU) to each token vector independently without cross-token communication.