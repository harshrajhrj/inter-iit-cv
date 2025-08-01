# Task Four
The task is divided into two phases:
1. Learning Goals
2. Project
## Learning Goals
#### Color Space Conversion
#### Blurring and smoothing
#### Edge Detection
#### Thresholding and contour detection
#### Drawing saped and text on image
#### Face detection
#### Object tracking
#### Masking techniques

#### Data Preprocessing steps
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

## Smart Attendance System (Task)
Download haarcascade face detection model from [opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades)
1. haarcascade_frontalface_default.xml
2. haarcascade_profileface.xml

Install the following package to import the face recognition model:<br>
```pip install opencv-contrib-python```

-----------
To train the model, follow the steps:
1. We can provide your own dataset in this hierarchy <br>
```
Faces |
	  |train |
	  |______|_name_1 |
	  |______|________|_img_1
	  |______|________|_img_2
	  |
	  |val___|
	  |______|_name_1 |
	  |______|________|_img_1
	  |______|________|_img_2
```
2. Run `preprocess.py`
3. Run `face_Train.py`

-----------
To test the trained model, follow the steps:
1. Open **face_recognition.py** and give an image path as <br>
	`img_name = <img_path>`
2. Run **face_recognition.py**.
	The result will be the face recognized image with face label predicted by the model. The confidence level of recognized is also printed on the console.