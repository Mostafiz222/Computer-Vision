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
1. Non-Uniform or Severe Directional Lighting (Shadows & Specular Glare)Why it fails: Otsu’s algorithm relies on a global intensity histogram assuming a bimodal distribution (one distinct peak for background, one for foreground).Shadows: A strong shadow across the surface lowers background pixel intensities into the same range as the coins. Otsu selects a threshold that misclassifies shadowed background areas as foreground, creating massive false-positive shapes that merge coins together.Specular Glare: Highly reflective metallic coins under direct top-lighting create bright white hotspots (overexposure). This collapses color/grayscale contrast, causing the centers or edges of coins to drop out during thresholding and resulting in broken, hollow, or ring-shaped contours.
2. Low Contrast Between Background and ObjectWhy it fails: Global thresholding requires a distinct separation between object and background gray levels.Similar Reflectance: If metallic coins are placed on a light metallic, gray, or reflective surface, the grayscale values of the coin body and the background overlap heavily.Histogram Collapse: The intensity histogram merges into a single unimodal distribution. Otsu fails to find an optimal variance minimum, leading to either an empty threshold mask (0 objects detected) or massive random noise artifacts across the image.
3. Object Occlusion, Clustering, and Variable ScalesWhy it fails: The pipeline makes two strict spatial assumptions: objects do not overlap, and objects fall within a strict pixel-area window ($\text{Area} \ge 2000$).Overlapping / Stacking: In realistic coin-sorting tasks, coins frequently overlap or pile up. Because cv2.findContours operates purely on pixel connectivity, overlapping coins form a single irregular blob.Scale Variance (Camera Distance): If the camera-to-table distance changes, the pixel area of a coin scales quadratically. Coins moved further away drop below the hardcoded 2000 area threshold and are filtered out, while non-coin background noise close to the camera gets detected as a coin.