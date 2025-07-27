# Task Four
The task is divided into two phases:
1. Learning Goals
2. Project
## Learning Goals
### Color Space Conversion
### Blurring and smoothing
### Edge Detection
### Thresholding and contour detection
### Drawing saped and text on image
### Face detection
### Object tracking
### Masking techniques
## Smart Attendance System (Task)
Key steps
- Image classification
- Object detection
- Image segmentation

### Basics (only includes important points)
#### Reading an image
- A large image may go off the screen due to its size. The dimensions of the image are way greater than the dimensions of the screen.
- We might get an error if the video goes out of frames or capture couldn't find the media file during `capture.read()`
### Data Preprocessing steps
Data preprocessing in computer vision tasks involves preparing raw image data for effective use by machine learning models. The key steps include: 
 - Data Cleaning: 
    - Handling Missing Data: Addressing incomplete image data or metadata. 
    - Noise Reduction: Removing irrelevant or corrupted information from images (e.g., speckle noise, salt-and-pepper noise). Techniques like median filtering or Gaussian blurring can be applied. 
	- Outlier Detection and Removal: Identifying and potentially removing images or features that significantly deviate from the norm, which could skew model training. 

 - Data Transformation: 
	- Image Resizing: Standardizing image dimensions to a consistent size required by the model. This is crucial for batch processing and model input requirements. 
	- Normalization: Scaling pixel values to a specific range (e.g., 0-1 or -1 to 1) to improve model convergence and performance. This often involves dividing pixel values by 255 for 8-bit images. 
	- Grayscale Conversion: Converting color images to grayscale if color information is not essential for the task, reducing computational complexity. 

 - Data Augmentation: 
	- Geometric Transformations: Applying operations like rotation, translation, scaling, flipping (horizontal/vertical), and shearing to create variations of existing images. This helps in increasing the dataset size and improving model generalization. 
	- Color Augmentations: Adjusting brightness, contrast, saturation, and hue to simulate different lighting conditions and improve robustness. 
	- Random Erasing/Cutout: Masking out random patches of an image to encourage the model to learn more robust features and reduce reliance on specific image regions. 

 - Feature Extraction (Optional but common): 
	- Manual Feature Engineering: In some traditional computer vision approaches, specific features like SIFT, HOG, or SURF are extracted from images. In deep learning, this is often handled automatically by convolutional layers. 

 - Data Splitting: 
	- Training, Validation, and Test Sets: Dividing the preprocessed dataset into distinct sets for model training, hyperparameter tuning and early stopping, and final performance evaluation, respectively. This ensures an unbiased assessment of the model's generalization capabilities.

