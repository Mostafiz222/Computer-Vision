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


#Lesson 4:DeiT (Data-efficient Image Transformer)
The Problem with ViT
The original ViT paper achieved remarkable results, but it had one major drawback:
It required hundreds of millions of images (JFT-300M) for effective training.
Most researchers and companies do not have access to datasets of this scale.
The question became:
Can Transformers be trained effectively using only ImageNet-1K (~1.3 million images)?
The answer was DeiT.
ViT
+
Better Training
=
DeiT

Why It Helps

The CNN teacher already captures useful visual patterns such as:
edges
textures
shapes
Instead of learning everything from scratch, the ViT benefits from this guidance.

Training Objective

The model optimizes two objectives:
Classification loss (ground-truth labels)
Distillation loss (teacher predictions)
Together they improve data efficiency.

#Lesson 5:Swin Transformer
Standard ViT computes self-attention over all image patches.
If an image has N patches:
Attention Complexity
O(N²)
For high-resolution images, this becomes computationally expensive.
Instead of attending globally,

Swin Transformer attends locally within windows.
Example:
Entire Image
↓
Split into Windows
↓
Self-Attention inside each Window

#Window-based Self-Attention
Imagine a 224×224 image divided into many windows.
+----+----+
| W1 | W2 |
+----+----+
| W3 | W4 |
+----+----+
Each window performs self-attention independently.
This greatly reduces computation.
| Feature                | ViT      | DeiT    | Swin              |
| ---------------------- | -------- | ------- | ----------------- |
| Global Attention       | ✅        | ✅       | ❌ (Local Windows) |
| Shifted Windows        | ❌        | ❌       | ✅                 |
| Hierarchical Features  | ❌        | ❌       | ✅                 |
| Data Efficient         | Moderate | High    | High              |
| High-Resolution Images | Limited  | Limited | Excellent         |

* **Why standard ViT is expensive:** Compute scales **quadratically** O(n^2) with the number of patches. Doubling image resolution increases attention memory/compute by 16 times.
* **How shifted windows work:** Alternating layers shift local window boundaries diagonally. This connects patches from adjacent prior windows, allowing information to spread across the full image in O(n) linear time.
* **Why Swin excels at detection/segmentation:** It builds a **hierarchical multi-scale feature pyramid** (1/4, 1/8, 1/16, 1/32), essential for detecting multi-size objects and mapping pixel-level boundaries.

#Lesson 6:CLIP (Contrastive Language–Image Pretraining)

Why Was CLIP Created?
Traditional image classifiers work like this:

Image
↓
Neural Network
↓
Dog

The problem is that the model can only predict classes it has been trained on.
If it has never seen "Red Panda" during training,
it cannot recognize it.
OpenAI asked a different question:
Can a model understand images using natural language?
#Instead of learning

Image
↓
Class ID

CLIP learns
Image
↔
Text
Core Idea

Suppose we have
Image of a dog
Caption
"A golden retriever playing."
CLIP learns that
Image Feature
≈
Text Feature
Images and text are mapped into the same embedding space.

Dual Encoder Architecture
Unlike ViT,
CLIP has two encoders.
           Image
              │
              ▼
      Vision Encoder
              │
      Image Embedding
              │
              │
              ▼
        Similarity
              ▲
              │
      Text Embedding
              │
        Text Encoder
              ▲
              │
          Caption

The vision encoder can be:
ResNet
ViT
The text encoder is usually a Transformer.

Why Is CLIP Important?

Instead of memorizing class IDs,

it learns visual semantics.

This makes it useful for:

Image search
Image retrieval
Zero-shot classification
Vision-language systems
Multimodal AI

* **Why two encoders?**
One processes visual inputs (images) and the other processes textual inputs (text prompts) so both modalities can be encoded into separate vector representations.
* **Why zero-shot classification works?**
Class labels are formatted as natural language prompts (e.g., *"a photo of a [dog]"*). CLIP compares the image embedding against all text label embeddings and selects the closest semantic match.
* **Why embeddings are in the same space?**
To enable direct distance comparison (via cosine similarity). Aligning image and text embeddings into a shared vector space allows the model to measure how closely a visual concept matches a textual description.

Lesson 7
DINO (Self-Distillation with No Labels)
The Motivation
Collecting labeled datasets is expensive.
Instead of using labels,
can a model learn useful visual representations directly from images?
DINO answers yes.

Supervised vs Self-Supervised

Traditional learning:

Image

↓

Cat
Needs labels.

DINO:

Image

↓

Representation
No labels are required.

Why Is DINO Popular?

It produces excellent feature representations.

Many downstream tasks use frozen DINO features because they transfer well.

* **Why DINO doesn't require labels:**
It uses **self-distillation (self-supervised learning)** where the model learns representations by predicting the output of one view of an image from another view of the same image, rather than fitting human-annotated class targets.
* **Teacher network's role:**
It generates stable target representations (pseudo-labels) for the student network to predict. Its weights are updated continuously as an **Exponential Moving Average (EMA)** of the student's weights to prevent collapsed representations.
* **Multiple image augmentations used:**
Passing different global (large region) and local (small crop) views forces the network to map both fine-grained local details and global scene context to the same semantic representation, teaching it invariant feature representations across scales and transformations.

#Lesson 8:DINOv2
Why Create DINOv2?

DINO was already strong,

but Meta wanted a model that could serve as a general-purpose visual foundation model.

Goals:

Better representations.
Larger training data.
Better scaling.
Better transfer across many tasks.

Major Improvements
1. Better Training Data

Meta curated a massive, high-quality dataset containing hundreds of millions of images.

The emphasis was on quality and diversity rather than simply collecting more images.

2. Larger Models

DINOv2 is available in multiple sizes:

Small
Base
Large
Giant

Larger models generally provide stronger representations at higher computational cost.

3. Better Feature Quality

Compared to DINO,

DINOv2 produces features that are more:

Robust
General
Transferable

This makes them useful across classification, retrieval, segmentation, depth estimation, and other tasks.

Why Researchers Like DINOv2

Suppose you freeze the backbone.

Even without fine-tuning,

a simple linear classifier often achieves very competitive performance.

This is why DINOv2 is frequently used as a feature extractor.

Architecture

The architecture is still based on Vision Transformers.

Image

↓

Patch Embeddings

↓

ViT Encoder

↓

Feature Vector

| Feature                      | CLIP                       | DINO                    | DINOv2                          |
| ---------------------------- | -------------------------- | ----------------------- | ------------------------------- |
| Uses Text                    | ✅                          | ❌                       | ❌                               |
| Uses Labels                  | Image–Text Pairs           | ❌                       | ❌                               |
| Self-Supervised              | Contrastive                | Self-Distillation       | Improved Self-Distillation      |
| Zero-Shot Classification     | ✅                          | ❌                       | ❌                               |
| Excellent Feature Extraction | ✅                          | ✅                       | ✅ (Stronger)                    |
| Typical Downstream Use       | Vision–Language, Retrieval | Representation Learning | General-Purpose Vision Backbone |


Key Takeaways
CLIP learns by aligning images and text in a shared embedding space, enabling zero-shot classification.
DINO learns visual representations without labels through self-distillation using teacher and student networks.
DINOv2 builds on DINO with larger-scale, higher-quality training and produces stronger, more transferable visual features.

#Lesson 9
SigLIP (Sigmoid Loss for Language-Image Pretraining):
Why Was SigLIP Created?
CLIP was revolutionary, but it has one important limitation.
Recall how CLIP is trained.
Suppose a batch contains
Dog
Car
Bird
CLIP compares every image with every text inside the batch.
This means training depends heavily on large batch sizes.
Large batches require:
many GPUs
huge memory
expensive hardware
Google asked:
Can we remove this dependency?
The answer became SigLIP.
CLIP's Loss
CLIP uses
Softmax Cross Entropy
Inside one batch
Image1
competes against
Text1
Text2
Text3
...

Every sample becomes a negative example for every other sample.
This creates competition inside the batch.
SigLIP's Idea

Instead of asking

Which text is the correct one among all texts?
SigLIP asks
Does this image match this text?
Only Yes or No.
That turns the problem into binary classification.
Sigmoid Loss
Instead of Softmax,
SigLIP uses
Sigmoid
Each image-text pair is evaluated independently.
Positive pair
Dog image
+
Dog caption
↓
Negative pair
Dog image
+
Car caption
↓
0
Every pair has its own probability.
No competition is needed.
Why Is This Better?
Since every pair is independent,
training no longer requires massive batches.
Advantages

✅ Smaller GPU memory
✅ Easier scaling
✅ Better training efficiency
✅ Often better retrieval performance

Architecture
The architecture is almost identical to CLIP.

Image
↓
Vision Encoder
↓
Embedding
↓
Similarity
↓
Text Encoder
↓
Text Embedding

The only major change is the loss function.

| Feature                | CLIP        | SigLIP      |
| ---------------------- | ----------- | ----------- |
| Vision Encoder         | ViT         | ViT         |
| Text Encoder           | Transformer | Transformer |
| Shared Embedding Space | ✅           | ✅           |
| Contrastive Learning   | ✅           | ✅           |
| Softmax Loss           | ✅           | ❌           |
| Sigmoid Loss           | ❌           | ✅           |
| Needs Very Large Batch | Yes         | No          |

One interesting question is:
Does changing the pretraining objective (Softmax vs Sigmoid) influence robustness under distribution shift?
That is a meaningful empirical research question because the representation learning objectives differ.

#Lesson 10:Linear Probing
What Is Linear Probing?
Suppose you download a pretrained DINOv2.
Should you retrain all 86 million parameters?
Usually,
No.
Instead,
freeze everything.
Train only one linear layer.
Image
↓
Frozen Backbone
↓
Feature Vector
↓
Linear Layer
↓
Prediction

Why Freeze the Backbone?
The backbone has already learned
edges
textures
shapes
semantics
object parts
We only need to learn
Feature
↓
Class
Linear Layer
Mathematically
Prediction
=
Wx+b
Nothing more.
No hidden layers.
No attention.
No convolution.
Just one matrix multiplication.
Why Researchers Use Linear Probing
Suppose
Model A
95%
Model B
90%
Both are frozen.
The better result means
Model A learned a stronger representation.
Therefore,
linear probing measures representation quality, not adaptation ability.
Pipeline
Dataset
↓
Frozen Model
↓
Extract Features
↓
Train Linear Layer
↓
Evaluate
Advantages
✅ Very fast
✅ Very stable
✅ Cheap
✅ Fair comparison between backbones

#Lesson 11:Fine-Tuning
Why Fine-Tune?
Sometimes,
linear probing is not enough.
Suppose the pretrained model was trained on
Dogs
Cars
People
Now you want
Medical Images
The features may not perfectly transfer.
So we allow the backbone to adapt.
Full Fine-Tuning
Image
↓
Entire Model
↓
Update ALL Parameters
Everything learns.
Advantages
✅ Highest possible accuracy
Disadvantages
❌ Expensive
❌ Slow
❌ High GPU memory
❌ Can overfit on small datasets

#Partial Fine-Tuning
Instead of updating everything
update only
Last Block
Classification Head
This is often a good compromise.
#Lesson 12:Parameter-Efficient Fine-Tuning (PEFT)
Motivation
Imagine a model has
300 Million Parameters
Updating everything
means
300 Million Gradients
Large memory.
Large GPU.
Long training.
Can we adapt the model
without updating everything?
PEFT says
Yes.
#Main Idea

Freeze the pretrained model.
Insert tiny trainable modules.
Only those modules learn.
Frozen Model
+
Small Trainable Modules
↓
New Task
Why Is PEFT Useful?
Instead of updating
100%
of parameters,
we update
0.1%
or
1%
depending on the PEFT method.

LoRA (Low-Rank Adaptation)

LoRA is the most widely used PEFT technique.

Instead of changing a weight matrix directly,

Original Weight

W

LoRA learns a low-rank update:

ΔW = A × B

where A and B are much smaller matrices than W.

The effective weight becomes

W + ΔW

During training:

W remains frozen.
Only A and B are updated.

This drastically reduces the number of trainable parameters.

Visual Intuition
Original Model
↓
Frozen
↓
Insert LoRA Layers
↓
Train Only LoRA
↓
Inference
↓
Original Weight + Learned Update

<!-- #Why LoRA Works -->

Many neural network updates have low intrinsic rank.
Instead of learning a full matrix,
a compact low-rank approximation often captures the necessary adaptation.

| Method              |     Speed | GPU Memory | Trainable Parameters |                Typical Accuracy |
| ------------------- | --------: | ---------: | -------------------: | ------------------------------: |
| Linear Probe        | Very Fast |   Very Low |             Very Low |                Moderate to High |
| Partial Fine-Tuning |    Medium |     Medium |        Low to Medium |                            High |
| Full Fine-Tuning    |      Slow |       High |                  All | Highest (often, but not always) |
| LoRA (PEFT)         |      Fast |        Low |             Very Low | Often close to full fine-tuning |


| Method        | Main Idea                                                |
| ------------- | -------------------------------------------------------- |
| LoRA          | Low-rank weight updates                                  |
| Adapters      | Small bottleneck layers inserted between existing layers |
| Prompt Tuning | Learn task-specific prompt embeddings                    |
| Prefix Tuning | Learn trainable prefixes for attention layers            |
Lesson 13
Adapter Tuning
Motivation

Suppose you have a pretrained ViT with 300 million parameters.

Instead of changing the original weights, what if we insert tiny neural networks inside the model?

Those tiny networks learn the new task.

Everything else remains frozen.

Main Idea

Instead of

Input

↓

Transformer Block

↓

Output

we modify the block into

Input

↓

Transformer

↓

Adapter

↓

Output

The adapter is a small bottleneck MLP.
Adapter Architecture
A typical adapter looks like
Input (768)
↓
Linear (768 → 64)
↓
ReLU / GELU
↓
Linear (64 → 768)
↓
Output
Instead of learning a huge transformation,
the adapter learns a compact correction.
Why a Bottleneck?
Example
Original dimension
768
Adapter hidden size
64
Parameter count becomes much smaller.
Instead of updating millions of weights,
only the adapter parameters are trainable.
Residual Adapter
Most adapters use a residual connection.
Output
=
Input
+
Adapter(Input)
The original representation is preserved, while the adapter adds task-specific information.
Advantages
✅ Very stable
✅ Original model remains unchanged
✅ Different adapters can be stored for different tasks
✅ Good performance on many downstream tasks
Disadvantages
❌ Slightly increases inference latency because extra layers are executed.
❌ More trainable parameters than LoRA in many configurations.
Where Is It Used?
Adapters were popular before LoRA and are still used in:
Vision Transformers
Multimodal models
Language models
Multi-task systems

Lesson 14
Prompt Tuning
Motivation
Suppose you have CLIP
Instead of changing the model,
what if you changed the input?
That is Prompt Tuning.
NLP Analogy
Original prompt
Translate English to French:
Hello
Prompt tuning learns a better prompt automatically.
Vision Version
Vision Transformers process tokens.
Normally
CLS
Patch1
Patch2
Patch3
Prompt Tuning inserts learnable prompt tokens.
Prompt1
Prompt2
Prompt3
CLS
Patch1
Patch2
These prompt vectors are trainable.
The backbone remains frozen.
Training
Only the prompt embeddings are updated.
Everything else is frozen.
Why Does It Work?
The prompt acts like an instruction.
Instead of modifying the model,
it modifies the information entering the model.
Advantages
✅ Extremely parameter-efficient
✅ Very small memory usage
✅ Easy to switch between tasks
Limitations
❌ Usually less effective than LoRA for difficult adaptation tasks.
❌ Performance depends on prompt design and optimization.
Research Example
Visual Prompt Tuning (VPT) is a well-known method for adapting Vision Transformers by learning only prompt tokens while freezing the backbone.
Lesson 15
Prefix Tuning
Prompt Tuning modifies the input.
Prefix Tuning goes one step further.
It modifies the attention mechanism itself.
Recall Self-Attention
Attention uses
Query
Key
Value
Prefix Tuning learns extra
Key
Value
vectors.
Modified Attention
Original
Q
K
V
Becomes
Q
Prefix K
Original K
Prefix V
Original V
The model now attends to both the learned prefixes and the original tokens.
Intuition
Imagine a teacher saying:
"Before reading the question, remember these hints."
Those hints are the learned prefixes.
Advantages
✅ More expressive than simple prompt tuning.
✅ Still updates only a tiny fraction of parameters.
Disadvantages
❌ More complex implementation.
❌ Less common in computer vision than LoRA or VPT.

Lesson 16
IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
IA³ takes a different approach.
Instead of adding layers or low-rank matrices,
it learns scaling vectors.
Main Idea
Suppose the feature vector is
[2.1, 0.4, 1.3]
IA³ learns
[0.8, 1.5, 0.3]
The output becomes
[1.68, 0.60, 0.39]
Each feature is amplified or suppressed.
Why Is It Efficient?
Instead of learning a large weight matrix,
IA³ learns only a few scaling coefficients.
Advantages
✅ Extremely few trainable parameters.
✅ Very low memory consumption.
Limitations
❌ Less flexible than LoRA because it cannot introduce entirely new transformations.

| Method                  | What is Learned?        | Extra Modules     | Typical Trainable Params | CV Usage | Strengths                                                 | Weaknesses                               |
| ----------------------- | ----------------------- | ----------------- | -----------------------: | :------: | --------------------------------------------------------- | ---------------------------------------- |
| **Linear Probe**        | Classifier head         | ❌                 |                    <0.1% |   ⭐⭐⭐⭐⭐  | Fastest, ideal for representation evaluation              | Cannot adapt backbone                    |
| **BitFit**              | Bias terms              | ❌                 |                    <0.1% |     ⭐    | Simplest possible adaptation                              | Limited capacity                         |
| **Prompt Tuning (VPT)** | Prompt tokens           | Prompt tokens     |                    ~0.1% |   ⭐⭐⭐⭐   | Excellent for ViTs, tiny memory footprint                 | May struggle on large domain shifts      |
| **Prefix Tuning**       | Attention prefixes      | Prefix K/V        |                ~0.1–0.5% |    ⭐⭐    | More expressive than prompt tuning                        | Less common in vision                    |
| **IA³**                 | Scaling vectors         | ❌                 |                    <0.1% |    ⭐⭐    | Extremely lightweight                                     | Less expressive than LoRA                |
| **Adapters**            | Small bottleneck MLPs   | ✅                 |                    ~1–5% |    ⭐⭐⭐   | Stable, supports multi-task adapters                      | Adds inference latency                   |
| **LoRA**                | Low-rank weight updates | Low-rank matrices |                  ~0.1–2% |   ⭐⭐⭐⭐⭐  | Excellent accuracy/efficiency trade-off, widely supported | Requires choosing target layers and rank |
| **Full Fine-Tuning**    | All parameters          | ❌                 |                     100% |   ⭐⭐⭐⭐⭐  | Maximum adaptation capacity                               | Highest memory and compute cost          |


Which Method Should You Choose?

| Goal                                    | Recommended Method           |
| --------------------------------------- | ---------------------------- |
| Evaluate representation quality         | Linear Probe                 |
| Very limited GPU memory                 | BitFit or IA³                |
| Vision Transformer adaptation           | LoRA or Visual Prompt Tuning |
| Multi-task deployment                   | Adapters                     |
| Best balance of accuracy and efficiency | LoRA                         |
| Maximum performance with ample compute  | Full Fine-Tuning             |
