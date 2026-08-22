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


#lesson3:Vision Transformer (ViT)

Why do we need Vision Transformers? 
CNNs have dominated Computer Vision for decades because they are excellent at learning local patterns such as edges, corners, and textures.
However, CNNs have some inherent limitations
They focus on local neighborhoods.
Capturing long-range relationships requires many convolutional layers.
Their inductive biases (locality and translation equivariance) make learning easier but can also limit flexibility.
Transformers remove these assumptions and allow every image region to interact with every other region from the very first layer.
#Step 1 — Split the Image into Patches
Suppose we have a 224 × 224 × 3 RGB image.
Choose a pa
Each side contains:
224 / 16 = 14 patches
Therefore:
14 × 14 = 196 patches
The image becomes
196 small images
instead of one large image.
#Step 2 — Flatten Each Patch
Each patch has
16 × 16 × 3
=768 values
Flatten it
Patch
↓
768-dimensional vector
Every patch now resembles a feature vector rather than a 2D image.
#Step 3 — Linear Projection
Instead of feeding the raw 768-dimensional vector to the Transformer, a learnable linear layer projects it into a fixed embedding size.
Example:
768
↓
768
or
768
↓
512
depending on the model.
These become the patch embeddings.
#Step 4 — Add Positional Embeddings

Transformers do not know where patches came from.

Therefore we add learnable position vectors.
Patch Embedding
+
Position Embedding
↓
Final Token
Now the model knows both:

What the patch contains.
Where it is located.
Step 5 — Add the Classification Token (CLS Token)
ViT introduces one extra learnable token.
[CLS]
Patch1
Patch2
...
Patch196

The CLS token collects information from all patches during self-attention.
After the last Transformer layer, only the CLS token is passed to the classification head.
Overall ViT Pipeline
Input Image
      │
      ▼
Split into Patches
      │
      ▼
Flatten
      │
      ▼
Linear Projection
      │
      ▼
Add Position Embeddings
      │
      ▼
Add CLS Token
      │
      ▼
Transformer Encoder × L
      │
      ▼
CLS Token
      │
      ▼
MLP Head
      │
      ▼
Prediction

Why ViT Works

Each patch attends to every other patch.

Example:

Car wheel

↓

can directly attend to

↓

Car window

without needing many convolutional layers.

The model naturally learns global object relationships.

Advantages

✅ Excellent scalability.

✅ Learns global context immediately.

✅ Works exceptionally well with very large datasets.

✅ Strong transfer learning performance.

Limitations

❌ Requires huge datasets.

❌ Training from scratch is expensive.

❌ Computational cost grows quadratically with the number of patches (O(N²) attention).

This limitation motivated the next generation of models.


