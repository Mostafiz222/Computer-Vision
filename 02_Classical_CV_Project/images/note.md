Project: Smart Coin Counter & Analyzer
Question 1
Why do we apply Gaussian Blur before Otsu thresholding instead of after?
->because Otsu's algorithm depends entirely on the pixel intensity histogram to calculate an optimal global threshold.
Question 2
Why is RETR_EXTERNAL a good choice for counting coins?
->because it retrieves only the extreme outer boundaries of objects, ignoring any nested inner contours.
Question 3
Suppose two coins are touching each other.
Will this pipeline count them correctly?
Why or why not?
->No, this pipeline will not count them correctly. It will group two touching coins together and count them as a single large coin.

Question 4 (Most Important)
Think like a researcher.
This pipeline works well on clean images.
What are three situations where it would fail in the real world?
Don't just list them—explain why they cause problems.

->Classical computer vision pipelines built on global thresholding and simple morphological filters are notoriously fragile when moving from controlled lab settings to real-world environments.Three primary failure modes in real-world deployments are detailed below.

1. Non-Uniform or Severe Directional Lighting (Shadows & Specular Glare)Why it fails: 
2. Low Contrast Between Background and ObjectWhy it fails: 
3. Object Occlusion, Clustering, and Variable ScalesWhy it fails