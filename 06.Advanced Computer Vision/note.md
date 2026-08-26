#Lesson 1: Contrastive Learning:
part1:The Limitation of Supervised Classification
Imagine a dataset of cats and dogs.
Traditional supervised learning trains a model like this:
Image
   │
CNN / ViT
   │
Feature Vector
   │
Classifier
   │
Cat or Dog
The model's objective is simply:
Predict the correct label.
After training, we usually discard the classifier and keep only the backbone.
The problem is that the backbone was optimized for one specific task.
If tomorrow you want to:
image retrieval
clustering
anomaly detection
zero-shot learning
transfer learning
the learned features may not be ideal.
Representation Learning
Instead of learning
Image → label
we want
Image → Meaningful Representation
A representation (embedding) is a numerical description of an image.
Example:
Cat
[0.13
-0.42
0.81
...
0.09]
Another cat
[0.15
-0.39
0.84
...
0.11]

Dog
[-0.77
0.62
-0.33
...
0.41]
Good representations satisfy:
Similar images → nearby vectors
Different images → far apart vectors
This is the core idea behind Contrastive Learning.
Part 2 — Embedding Space
Why is this powerful?
Once embeddings are good,
you can perform:
nearest neighbor search
clustering
retrieval
few-shot learning
transfer learning
anomaly detection
without retraining a classifier.

#The Core Idea of Contrastive Learning
The idea is surprisingly simple.
The model repeatedly answers one question:
Which samples should be close together, and which should be far apart?
Positive Pair
Two images representing the same semantic content.
Example:
Original Cat
↓
Random Crop
↓
Another Crop
↓
Color Jitter
↓
Blur
All are still the same cat.
These become
Positive Pair.
Negative Pair
Different semantic content.
Example
Ca
Dog
Car
Tree
Airplane
These should be separated.
Goal
Positive Pair
Bring Together
↓
Embedding Distance ↓
Negative Pair
Push Apart
↓
Embedding Distance ↑
Visual Intuition
Initially
Cat1
Dog
Cat2
Car
Bird
Everything mixed.
After training
Cats
● ● ●
Dogs
▲ ▲
Cars
■ ■
Birds
◆ ◆
The model organizes the feature space automatically.

#Why CLIP Uses Contrastive Learning
CLIP doesn't compare:
Image ↔ Image
Instead it compares:
Image
↓
Embedding
↑
↓
Embedding
Text
Example
Image
"A dog running"
↓
close
Text
"A dog running"
but
Image
"A dog"
↓
far
Text
"A car"
By aligning image and text embeddings, CLIP learns a shared embedding space where related images and captions are close together.

#Real Research Applications
Contrastive Learning powers:
CLIP
SigLIP
SimCLR
MoCo
DINO
DINOv2
BLIP
ALIGN
It is used for:
Image Search
Visual Search Engines
Image Retrieval
Self-Supervised Learning
Vision-Language Models
Robotics
Medical Imaging
Autonomous Driving

**Representation Learning vs. Classification**
Classification trains a model to predict a fixed set of labels (e.g., mapping an image strictly to "cat" or "dog"). Representation learning teaches the model to capture the underlying structure and features of data in a general feature space without tying it to specific labels. This makes the learned representations reusable across many downstream tasks.

**Embeddings**
An embedding is a dense, continuous vector representation of data (like text, images, or audio) in a low-dimensional space, where semantic similarity corresponds to spatial proximity.

**Positive and Negative Pairs**

* **Positive Pair:** Two views or samples that share the same underlying content or semantic meaning.
* *Example 1:* An image of a cat and a cropped/color-jittered version of the same image.
* *Example 2:* An image of a beach and its accompanying text caption "A sunny day at the beach."


* **Negative Pair:** Two distinct samples that represent different concepts or entities.
* *Example 1:* An image of a cat and an image of a sports car.
* *Example 2:* An image of a dog and the text caption "A plate of delicious sushi."



**Importance of Data Augmentations**
Augmentations prevent the model from learning trivial shortcuts (like matching pixel colors or background noise). By forcing the model to recognize that different transformed versions of the same image belong together, it learns robust, semantic invariant features.

**Why CLIP Uses Image-Text Pairs**
Comparing images with text allows CLIP to learn rich, natural language concepts instead of rigid, discrete categories. This multimodal alignment enables zero-shot transfer to almost any vision task without needing task-specific retraining.

**Answers to Challenge Questions**

**False Negatives (Class Collapse / False Negative Penalty)**
Treating two distinct dog images as a negative pair forces the model to push their embeddings far apart, even though they share similar semantic features. This pushes valid semantic clusters apart and can degrade representation quality.

**Limits of Random Rotation**
Random rotation cannot be used when spatial orientation carries explicit semantic meaning.

* *Example:* Digit recognition (e.g., rotating a **6** by $180^\circ$ turns it into a **9**, altering its ground-truth identity).

**Transferability of Contrastive Learning**
Classification models focus heavily on learning features relevant only to distinguishing pre-defined classes (often discarding subtle background or fine-grained detail). Contrastive learning forces the network to capture richer, structural representations of the entire input space, making those features far more adaptable to new, unseen downstream tasks.

#SimCLR — The First Modern Contrastive Learning Framework
Suppose you have 1 million unlabeled images.
Traditional supervised learning cannot train directly because there are no labels.
The question becomes:
Can a model learn meaningful visual representations without knowing the object classes?
SimCLR answered:
Yes. Let the data supervise itself.
Instead of using labels, SimCLR creates its own learning signal using data augmentation.

The SimCLR Pipeline

The complete pipeline is surprisingly simple.

               Original Image
                     │
        ┌────────────┴────────────┐
        │                         │
 Augmentation A             Augmentation B
        │                         │
        ▼                         ▼
      View 1                   View 2
        │                         │
        └────────────┬────────────┘
                     │
               Same Encoder
                (Shared Weights)
                     │
        ┌────────────┴────────────┐
        │                         │
      h₁ (Feature)             h₂ (Feature)
        │                         │
        ▼                         ▼
     Projection Head         Projection Head
        │                         │
      z₁ (Embedding)          z₂ (Embedding)
        │                         │
        └────────────┬────────────┘
                     │
                InfoNCE Loss

#Strengths and Weaknesses
Strengths
No manual labels required.
Learns high-quality transferable representations.
Simple architecture.
Strong downstream performance.
Influenced nearly all later contrastive methods.
Weaknesses
Requires very large batch sizes.
Heavy GPU memory usage.
Training is computationally expensive.
Performance depends strongly on augmentation quality.
These limitations motivated later methods such as MoCo, which replaced large batches with a memory queue.
Real-World Applications

SimCLR-style pretraining is used in:

Medical image representation learning
Satellite image analysis
Industrial defect inspection
Robotics perception
Autonomous driving
Foundation model pretraining

Even if newer methods are adopted, many still build upon SimCLR's core ideas.

**SimCLR Architecture & Training Mechanics**

* **Two Augmented Views:** Creating two distinct augmented views (x_i, x_j) from the same image provides a positive pair for the contrastive loss to pull together, forcing the model to learn representations invariant to those specific transformations.
* **Shared Encoder Weights:** Sharing weights ensures that both views are mapped into the exact same feature space. If separate parameters were used, the networks could solve the task trivially by encoding branch-specific shortcuts rather than true semantic features.
* **Projection Head:** The non-linear projection head $g(\cdot)$ maps representations to a lower-dimensional space where the contrastive loss is applied. It acts as a buffer that removes transformation-specific noise from the base feature representations.
* **Discarding Projection Head:** After pretraining, the projection head is removed because it discards task-agnostic feature details in favor of optimizing the contrastive loss. The representations from the backbone $h = f(x)$ retain richer, more adaptable information for downstream tasks.
* **Importance of Large Batch Sizes:** SimCLR relies exclusively on in-batch samples to source negative pairs (a batch size $N$ yields $2(N-1)$ negative pairs). Larger batches provide a wider variety of negative samples, making the contrastive task harder and stabilization smoother.
* **Cosine Similarity:** Cosine similarity measures the angle between vectors rather than their magnitude, assessing directional alignment. Normalizing features onto a unit hypersphere prevents the network from inflating vector lengths to minimize loss.

---

**Challenge Questions**

**Batch Size Limitations ($N=32$)**
With a batch size of 32, the model only sees 62 negative pairs per positive pair. This sparse pool of negatives makes the contrastive optimization task too easy, causing gradients to become noisy and performance/representation quality to drop significantly.

**Augmentation Intensity Effects**

* **Too Weak:** The model learns trivial visual shortcuts (like matching global color distributions or spatial layouts) rather than high-level semantic features, leading to feature collapse or poor generalization.
* **Too Aggressive:** Key semantic content is destroyed (e.g., cropping out the main object entirely), forcing the model to push apart views that actually share identical semantic context, which corrupts the representation space.

**SimCLR vs. Supervised Classifier Transferability**
Supervised models discard any input information unnecessary for discriminating between their fixed training classes. SimCLR's instance discrimination task forces the backbone to preserve broad structural, textural, and contextual details, making those features far more versatile when adapting to completely new downstream targets.

#Lesson 3: Self-Supervised Learning (SSL)
Why Self-Supervised Learning?

Deep learning has traditionally relied on large labeled datasets.

For example:

Dataset	Images	Labels Required
MNIST	70K	Yes
CIFAR-10	60K	Yes
ImageNet	14M+	Yes
Creating these labels is:
Expensive
Time-consuming
Error-prone
Difficult in specialized domains (e.g., medicine, satellite imagery)
Meanwhile, billions of unlabeled images already exist.
The central question becomes:
Can a model learn useful visual representations without human annotations?
SSL answers yes.

The Three Learning Paradigms
1. Supervised Learning
Image
   ↓
Human Label
   ↓
Model
Example:
Dog Image → Dog
Needs human annotation.

2. Unsupervised Learning
No labels.
Typical tasks include:
Clustering
Dimensionality reduction
Density estimation
The model discovers patterns but is not directly optimized for downstream prediction tasks.

3. Self-Supervised Learning

No human labels.

Instead, the data itself generates the supervision signal.
Example:
Original Image
↓
Create two augmented views
↓
Learn that both views belong together
No human labeled the image.
The supervision comes from the image itself.
What Is a Pretext Task?
A pretext task is an automatically generated task whose purpose is not the final application, but learning useful representations.
Think of it as practice before the real exam.
Examples:
Predict missing image patches
Match two augmented views
Predict image rotation
Match image and text
Reconstruct masked regions
The downstream task might later be:
Classification
Detection
Segmentation
Retrieval

#SSL Training Pipeline

A typical SSL workflow is:
Large Unlabeled Dataset
          │
          ▼
Self-Supervised Pretraining
          │
          ▼
Learn General Representations
          │
          ▼
Fine-tune or Linear Probe
          │
          ▼
Downstream Task
This is exactly how many modern Vision Foundation Models are trained.

#Real-World Applications

SSL is widely used in:

Medical imaging (limited expert labels)
Satellite imagery
Robotics
Autonomous driving
Manufacturing defect detection
Video understanding
Vision-language models

Pipeline (1M Unlabeled + 1K Labeled)

SSL Pre-training: Train a backbone (e.g., SimCLR, MAE) on the 1M unlabeled images using contrastive learning or masked reconstruction to extract general visual features.

Supervised Fine-Tuning: Freeze or low-learning-rate fine-tune the encoder on the 1K labeled images with a linear classification head.

Can SSL Replace Supervised Learning?

No. SSL learns general representations, but final task-specific mapping, class definition, evaluation, and fine-grained alignment still require labeled ground truth.

Why SSL Transfers Better

No Label Bias: Supervised models throw away features irrelevant to target labels (e.g., background context). SSL retains overall visual structure.

Fewer Shortcuts: Prevents the model from over-indexing on simple visual shortcuts (like background color) to predict a category.

Richer Representations: Pretext tasks force the network to encode spatial, textural, and structural visual geometry across the entire image distribution.

#Lesson 4: Domain Adaptation
What is a Domain?
A domain is the environment or distribution from which data comes.
A domain is defined by:
Input data distribution
Image characteristics
Sensor type
Lighting
Background
Weather
Camera properties

| Dataset          | Domain                             |
| ---------------- | ---------------------------------- |
| CIFAR-10         | Small natural images               |
| CIFAR-100        | Small natural images (100 classes) |
| ImageNet         | High-resolution internet images    |
| Chest X-rays     | Medical imaging                    |
| Satellite Images | Remote sensing                     |
| Thermal Images   | Infrared imaging                   |


Why Does Domain Shift Hurt Performance?
A model learns statistical patterns from the training data.
For example:
During training:
Cats usually appear on bright backgrounds.
During testing:
Cats appear in low light.
The learned representation may no longer match the new distribution.
As a result:
Accuracy drops
Confidence decreases
Predictions become unreliable
This is one of the biggest challenges in deploying AI systems.

What is Domain Adaptation?
Domain Adaptation aims to transfer knowledge from one domain (source) to another (target).
Source Domain
(Labeled)
↓
Train Model
↓
Adapt
↓
Target Domain
(Few or No Labels)

Types of Domain Adaptation
1. Supervised Domain Adaptation

Target domain has labels.

Example:

Source: Hospital A (labeled)
Target: Hospital B (also labeled)

The target labels are available during adaptation.

2. Semi-Supervised Domain Adaptation

Target domain has only a small number of labels.

Example:

50,000 labeled source images
500 labeled target images

The model uses both labeled and unlabeled target data.

Unsupervised Domain Adaptation (UDA)

The most common research setting.

Source:
Labeled
Target:
Unlabeled
Goal:
Learn a model that performs well on the unlabeled target domain.

How Domain Adaptation Works (High Level)

Different methods exist, but most aim to make source and target features similar.
Source Images
      │
      ▼
 Feature Extractor
      │
      ▼
 Feature Space
      ▲
      │
Target Images

Domain Adaptation vs Transfer Learning
| Transfer Learning                  | Domain Adaptation                      |
| ---------------------------------- | -------------------------------------- |
| Reuse pretrained model             | Adapt to a different data distribution |
| Usually fine-tune on target labels | Often has few or no target labels      |
| Focus on task transfer             | Focus on distribution shift            |

Real-World Applications

Domain Adaptation is widely used in:

Autonomous driving
Medical imaging across hospitals
Satellite imagery across regions
Manufacturing inspection
Agricultural drones
Security cameras with different lighting
Robot vision across environments

Challenge Questions
1.You train a crop disease detector using drone images from Bangladesh and deploy it in Brazil. What types of domain shift might occur?
2.Why is unsupervised domain adaptation often considered more practical than supervised domain adaptation?
3.If source and target domains are extremely different (e.g., cartoons → X-rays), would domain adaptation still be effective? Why or why not?

**1. Domain Shift (Bangladesh → Brazil)**

* **Covariate Shift:** Visual differences in soil, lighting, drone altitude, leaf angles, and crop varieties.
* **Prior Shift:** Disease prevalence differs due to climate and regional farming practices.
* **Concept Shift:** Lookalike symptoms (e.g., heat stress in Brazil looking like a fungal infection from Bangladesh).

**2. Why Unsupervised DA (UDA) is More Practical**

* **Zero Label Cost:** Raw target data is free to collect, whereas target labels require expert manual annotation (e.g., agronomists).
* **Autonomous Deployment:** Systems can self-adapt on the fly without waiting for human labeling workflows.

**3. Extreme Shift (Cartoons → X-rays)**
**No, domain adaptation will fail.** DA assumes shared structural and semantic feature spaces. Cartoons and X-rays share almost no common textures, geometry, or semantics, causing **negative transfer** where distribution alignment degrades model performance.

Lesson 5: Domain Generalization
Imagine you train an image classifier using photos collected in:

Bangladesh
India
Nepal
Now you deploy it in:
Brazil
You have never seen any Brazilian data during training.
Can the model still perform well?
This is exactly the goal of Domain Generalization.

Domain Generalization
During training:
Source Domain 1 ✓
Source Domain 2 ✓
Source Domain 3 ✓
Target Domain ✗
The target domain is completely unseen.
The model must generalize without ever observing target data.

#Meta-Learning
Training simulates domain shift.
Example:
Train using:
Hospital A
↓
Validate on Hospital B
Then switch:
Hospital B
↓
Validate on Hospital C
The model repeatedly practices adapting to unseen domains.

#Domain Generalization Pipeline
Multiple Source Domains
          │
          ▼
Feature Extractor
          │
          ▼
Domain-Invariant Representation
          │
          ▼
Classifier
          │
          ▼
Unseen Target Domain

| Aspect                  | Transfer Learning | Domain Adaptation | Domain Generalization |
| ----------------------- | ----------------- | ----------------- | --------------------- |
| Target data available?  | Usually yes       | Yes               | No                    |
| Target labels required? | Usually yes       | Often no          | No                    |
| Main challenge          | New task          | Domain shift      | Unseen domain         |
| Goal                    | Reuse knowledge   | Adapt             | Generalize            |
Real-World Applications

Domain Generalization is important in:

Medical imaging across unseen hospitals
Autonomous driving in new cities
Agricultural monitoring across countries
Wildlife monitoring
Remote sensing
Security surveillance
Industrial inspection


#Challenge Questions
1.If a model is trained only on sunny-day images, what failures might occur when deployed in snow or heavy rain?
2.Why can increasing dataset diversity improve domain generalization?
3.If a model achieves 99% accuracy on its training domains but performs poorly on a new domain, what does that tell you about the learned representations?

* **Sunny-day model failures:** It will likely misclassify or miss objects entirely due to **domain shift**. It cannot handle the altered lighting, reduced contrast, and new textures (like snow or rain streaks) because it overfit to sunny conditions.
* **Diversity and generalization:** Exposing the model to varied environments forces it to learn the **invariant, core features** of an object (its shape and semantic structure) rather than relying on domain-specific background cues.
* **99% train accuracy vs. poor new domain:** The model overfit to the source domains and learned **spurious correlations** ("shortcuts") rather than robust, general-purpose representations.

#Lesson 6: Out-of-Distribution (OOD) Detection

Imagine you train a classifier on CIFAR-10.

The classes are:

Airplane
Car
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
Now suppose you give it this image:
🥕 A carrot
The model has never seen carrots before.
Ideally, it should say:
"I don't know."
Instead, many deep learning models confidently predict:
Truck (99.2%)
This is the fundamental OOD problem.

What is OOD Detection?
OOD detection adds another decision before trusting the classifier.
Input Image
      │
      ▼
Feature Extractor
      │
      ▼
OOD Detector
      │
 ┌────┴────┐
 │         │
ID       OOD
 │         │
 ▼         ▼
Classifier Reject
If the sample is OOD,
the system refuses to make a prediction.

Common OOD Detection Methods
->Confidence Threshold
->Feature Distance
->Energy-Based Methods
->Foundation Model Features

Evaluation Metrics

Unlike classification,

OOD detection uses different metrics.

Common ones include:

AUROC (Area Under the ROC Curve)
AUPR (Area Under the Precision–Recall Curve)
FPR95 (False Positive Rate when TPR = 95%)

General interpretation:

Higher AUROC → better separation between ID and OOD.
Lower FPR95 → fewer OOD samples mistakenly accepted.

Real-World Applications

OOD detection is critical in:

Autonomous vehicles
Medical diagnosis
Industrial inspection
Airport security
Agricultural robotics
Financial fraud detection
Space exploration

Whenever a wrong prediction could be dangerous,

the model should know when not to answer.

#Lesson 7: Robustness
A model is robust if it continues to perform well when the input changes in ways that do not change the underlying semantic meaning.

Example:
Original image:
🐱
Modified image:
Slight blur
Rain
Camera noise
Lower brightness
Humans still recognize:
Cat
A robust model should also predict:
Cat

Why Accuracy is Not Enough
Suppose two models achieve:
Model	CIFAR-100 Accuracy
Model A	91%
Model B	90%
Now evaluate on corrupted images.
Model	CIFAR-100-C
Model A	43%
Model B	68%
Although Model A has higher clean accuracy, Model B is much more reliable in realistic conditions.
This is why robustness evaluation has become a standard part of computer vision research.

Types of Distribution Shift
1. Corruption Shift
Examples:
Gaussian Noise
Motion Blur
Fog
Snow
Brightness
Contrast
JPEG Compression
The object remains the same.
2. Natural Shift
Examples:
Different cameras
Different countries
Different weather
Different seasons
3. Synthetic Shift
Generated through:
Style transfer
Image editing
Simulation engines
4. Adversarial Shift
Inputs are intentionally modified to fool the model.
Unlike corruption, these perturbations are designed specifically to cause incorrect predictions.

Robustness Evaluation Pipeline
Clean Dataset
      │
      ▼
Train / Load Model
      │
      ▼
Evaluate on Clean Images
      │
      ▼
Evaluate on Corrupted Images
      │
      ▼
Compare Performance Drop

Real-World Applications

Robustness matters in:

Self-driving vehicles
Medical diagnosis
Agricultural drones
Industrial quality inspection
Surveillance systems
Space robotics
Military reconnaissance
In these settings, lighting, weather, sensor quality, and environmental conditions constantly change.

<!-- Lesson 8: Explainability (XAI) -->

Why Explainability Matters

Imagine a medical AI predicts:

Cancer

The doctor asks:

"Why?"
If the AI cannot explain its reasoning, the prediction is difficult to trust.
The same applies to:
Autonomous driving
Medical diagnosis
Industrial inspection
Financial systems
Security applications

High accuracy alone is often not sufficient.

The Black Box Problem

Deep neural networks contain millions or even billions of parameters.

Image
   │
Deep Neural Network
(100M+ Parameters)
   │
Prediction

We know the input.
We know the output.
But we often do not know why a specific prediction was made.
This is known as the black-box problem.

Interpretability vs Explainability
These terms are related but not identical.
Interpretability
A model is inherently understandable.
Example:
Decision Tree
Linear Regression
You can directly inspect how predictions are made.
Explainability
The model itself remains complex,
but we generate explanations after training.
Example:
Grad-CAM
SHAP
LIME
Attention visualization
Most modern deep learning models require explainability methods.

#Saliency Maps

A saliency map highlights pixels that most influence the prediction.
Example:
Input Image
↓
Compute Gradient
↓
Importance Map
Bright regions:
→ More important
Dark regions:
→ Less important
For a cat classifier,
the model should ideally focus on:
Eyes
Face
Ears
rather than:
Sky
Grass
Background

-->Gradient-weighted Class Activation Mapping (Grad-CAM) is one of the most widely used explainability techniques for CNNs.
Pipeline:

Input Image
      │
      ▼
CNN
      │
      ▼
Last Convolution Layer
      │
      ▼
Gradients
      │
      ▼
Heatmap
      │
      ▼
Overlay on Image

Example:
Dog Image
↓
Heatmap
🔥🔥🔥 Face
🔥 Body
Background (low importance)
The heatmap reveals which image regions contributed most to the prediction.

Popular Explainability Methods

| Method                 | Typical Use                                      |
| ---------------------- | ------------------------------------------------ |
| Saliency Maps          | Pixel importance                                 |
| Grad-CAM               | CNN heatmaps                                     |
| Guided Backpropagation | Fine-grained visualization                       |
| SHAP                   | Feature attribution                              |
| LIME                   | Local explanation                                |
| Attention Maps         | Transformer visualization                        |
| Integrated Gradients   | Attribution with improved theoretical properties |

Limitations of Explainability

Explainability methods are approximations, not ground truth.

Common challenges:

Different methods may produce different explanations.
Heatmaps can be unstable.
Some methods are sensitive to small input changes.
Attractive visualizations do not necessarily imply correct reasoning.
Explanations can sometimes highlight correlated background features instead of the true object.

Always validate explanations rather than assuming they are correct.

#Real-World Applications

Explainability is widely used in:

Medical imaging
Drug discovery
Autonomous driving
Industrial defect detection
Satellite image analysis
Regulatory AI systems
Scientific research

It helps developers:

Debug models
Detect shortcut learning
Build user trust
Meet regulatory requirements


<!-- Lesson 9: Object Detection -->
Image Classification tells us what is in an image. Object Detection tells us what is present and where it is.
Why Object Detection?

Suppose we have this image:

        🚗      🚗

    🚶

             🚲

A classification model predicts:

Car

Is that enough?

No.

We don't know:

How many cars exist?
Where are they?
Is there a pedestrian?
Is there a bicycle?

Object Detection answers all of these questions simultaneously.

Classification vs Localization vs Detection

These three tasks are often confused.

1. Image Classification

Input:

Image

Output:

Dog

Only predicts the image label.

2. Object Localization

Input:

Image

Output:

Dog

Bounding Box

Only one object is assumed.

Example:

+-------------------+
|                   |
|     ┌───────┐     |
|     │  Dog  │     |
|     └───────┘     |
|                   |
+-------------------+
3. Object Detection
Input:
Image
Output:
Dog
Bounding Box
Cat
Bounding Box
Person
Bounding Box
Multiple objects.
Multiple classes.
Multiple locations.
 What is a Bounding Box?
A bounding box is a rectangle surrounding an object.
Example
+---------------------------+
|                           |
|        ┌──────────┐       |
|        │   Dog    │       |
|        └──────────┘       |
|                           |
+---------------------------+

Usually represented as
(xmin, ymin, xmax, ymax)
or
(center_x,
 center_y,
 width,
 height)
Example
(120, 45, 280, 190)
These four numbers describe the object's location.
Detection Output

Suppose an image contains

one dog
two people
one bicycle
The detector may output
Class	Confidence	Bounding Box
Dog	0.98	(45, 30, 180, 210)
Person	0.95	(250, 40, 340, 240)
Person	0.91	(360, 35, 440, 245)
Bicycle	0.88	(180, 220, 420, 420)
Notice that every prediction contains:
Class
Confidence score
Location

#The Object Detection Pipeline
A modern detector follows this pipeline:
Input Image
      │
      ▼
Backbone Network
      │
Extract Features
      │
      ▼
Detection Head
      │
      ▼
Bounding Boxes
Class Labels
Confidence Scores

Backbone

Extracts visual features.

Examples:

ResNet
ConvNeXt
Vision Transformer
CSPDarkNet
Detection Head

Predicts

Object class
Bounding box coordinates
Confidence

Popular Object Detection Models

Object detectors are broadly divided into two families.

Two-Stage Detectors

Examples:

R-CNN
Fast R-CNN
Faster R-CNN
Mask R-CNN

Pipeline

Generate candidate regions

↓

Classify each region

Advantages

High accuracy
Strong localization
Disadvantages
Slower inference
One-Stage Detectors
Examples
YOLO
SSD
RetinaNet
Pipeline
Image
↓
Directly predict
boxes + classes

Advantages
Extremely fast
Real-time detection
Disadvantages
Earlier versions sacrificed some accuracy (modern YOLO versions have narrowed this gap significantly).
We'll study both families in later lessons.

#Evaluation Metrics

Unlike classification, detection must evaluate both:

Correct class
Correct location

The main metrics are:

IoU (Intersection over Union) – Measures how well the predicted box overlaps the ground-truth box.
Precision – Fraction of predicted detections that are correct.
Recall – Fraction of ground-truth objects that were detected.
Average Precision (AP) – Area under the Precision–Recall curve for one class.
Mean Average Precision (mAP) – Mean AP across all classes.

#Real-World Applications

Object Detection is used in:

Autonomous driving
Traffic monitoring
Face detection
Medical imaging
Drone surveillance
Manufacturing inspection
Wildlife monitoring
Retail inventory management
Robotics

<!-- Image Segmentation -->

Image Classification tells us what is in an image.
Object Detection tells us what and where.
Image Segmentation tells us the exact shape of every object at the pixel level.
Why Segmentation?

Suppose we have this image:

        🚗

   🚶

           🌳
Classification

Output

Car

Very little information.

Object Detection

Output

Car
Bounding Box

Person
Bounding Box

Tree
Bounding Box

Better.

But we still don't know the exact object boundaries.

Segmentation

Output

Every pixel belongs to a class.

Now the model knows
Exact object shape
Exact object boundary
Exact object size
This is much richer information.

What is a Mask?

Instead of predicting rectangles,

segmentation predicts a mask.

Example

Original

□□□□□□□
□□🐱□□□
□□🐱□□□
□□□□□□□
Mask
0000000
0011000
0011000
0000000
Each pixel receives a label.

Types of Image Segmentation

There are three major types.

1. Semantic Segmentation

Every pixel is assigned a class.

Example

Road
Road
Road

Car
Car

Sky
Sky

If there are three cars,

all car pixels receive

Car

The model does not distinguish individual objects.

Visual Example
🚗 🚗 🚗

↓

All become

CAR

Every car has the same label.

2. Instance Segmentation

Now each object receives its own mask.

Example

Car 1

Car 2

Car 3

Although all are cars,

the model separates them individually.

Output

Car #1

Car #2

Car #3

This is much more informative.

3. Panoptic Segmentation

Combines both tasks.

Things (countable)

Car
Person
Dog

Stuff (uncountable)

Sky
Road
Grass

Output

Road

Sky

Car #1

Car #2

Person #1

Today,

Panoptic Segmentation is considered one of the most complete scene understanding tasks.

#Detection vs Segmentation

| Task                  | Output                         |
| --------------------- | ------------------------------ |
| Classification        | Image Label                    |
| Localization          | One Bounding Box               |
| Object Detection      | Multiple Bounding Boxes        |
| Semantic Segmentation | Pixel Class Labels             |
| Instance Segmentation | Separate Mask for Every Object |
| Panoptic Segmentation | Instance + Semantic Together   |

Popular Segmentation Models
1. FCN (Fully Convolutional Network)

Published in 2015, FCN replaced fully connected layers with convolutional layers, allowing a network to produce dense pixel-wise predictions.

Pros

First modern segmentation architecture
End-to-end training

Cons

Coarse object boundaries
2. U-Net

Probably the most famous segmentation architecture.

Architecture

Encoder

↓

Bottleneck

↓

Decoder

The key innovation:

Skip Connections

Encoder Layer

──────────────►

Decoder Layer

These connections preserve fine spatial details.

Applications

Medical imaging
Microscopy
Satellite imagery
Industrial inspection
3. DeepLab

DeepLab introduced Atrous (Dilated) Convolutions, which enlarge the receptive field without reducing resolution.

Advantages

Better multi-scale context
Strong segmentation accuracy

Variants

DeepLabV1
DeepLabV2
DeepLabV3
DeepLabV3+
4. Mask R-CNN

Extends Faster R-CNN.

Pipeline

Detect Object

↓

Predict Bounding Box

↓

Predict Pixel Mask

Produces

Class
Bounding box
Segmentation mask

One of the most influential instance segmentation models.

5. Segment Anything Model (SAM)

Developed by Meta AI.

Instead of training for one dataset,

SAM is trained to segment almost any object from prompts.

Example prompts:

Click on an object
Draw a box
Provide a text prompt (with extensions)

SAM is considered a foundation model for segmentation.


Evaluation Metrics

Unlike classification,

segmentation evaluates predictions at the pixel level.

Pixel Accuracy

Percentage of correctly classified pixels.

Simple,

but can be misleading if one class dominates the image.

Intersection over Union (IoU)

Measures overlap between predicted and ground-truth masks.

Prediction ∩ Ground Truth
-------------------------
Prediction ∪ Ground Truth

IoU ranges from 0 to 1.

Higher is better.

Dice Score

Also called the F1 Score for segmentation.

Formula

2 × Intersection
-------------------------
Prediction + Ground Truth

Widely used in

Medical imaging
Organ segmentation
Tumor segmentation

Dice is especially useful when objects occupy only a small part of the image.

Part 7 — Real-World Applications

Segmentation is essential in:

Autonomous driving
Medical diagnosis
Satellite land-cover mapping
Precision agriculture
Robot navigation
Construction monitoring
Flood detection
Brain MRI analysis
Cell segmentation
Manufacturing inspection