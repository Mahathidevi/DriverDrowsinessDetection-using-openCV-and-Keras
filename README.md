# DriverDrowsinessDetection-using-openCV-
This code is a Python script that uses computer vision techniques and deep learning to monitor a person's eyes for drowsiness detection. It uses OpenCV, Keras, and Pygame libraries.

*******Step wise explanation of the code in Driver Drowsiness Detection.py file********

1. **Imports:**
   - `cv2`: OpenCV library for computer vision tasks.
   - `os`: Module for interacting with the operating system, used for file paths.
   - `keras.models`: Used for loading a pre-trained deep learning model.
   - `numpy as np`: NumPy for numerical operations.
   - `pygame.mixer`: Used to play an alarm sound.
   - `time`: For time-related functions.

2. **Initializing Mixer and Loading Sound:**
   - The code initializes the mixer for Pygame and loads an alarm sound from a file called 'alarm.wav'.

3. **Cascade Classifiers:**
   - Haar cascade classifiers are trained models for detecting specific objects, such as faces and eyes.
   - `face`, `leye`, and `reye` are instances of these classifiers, trained for detecting faces and left and right eyes.

4. **Labels and Model Loading:**
   - `lbl` is a list with two labels: 'Close' and 'Open'.
   - A pre-trained deep learning model is loaded using Keras for eye state classification. The model is loaded from 'cnnCat2.h5'.

5. **OpenCV Video Capture:**
   - OpenCV captures video from the default camera (Camera 0).
   - A font for text display is defined.

6. **Variables Initialization:**
   - `count`, `score`, and `thicc` are initialized to keep track of various metrics.
   - `rpred` and `lpred` are initialized to hold predictions for the right and left eyes.

7. **Main Loop:**
   - The code enters a main loop to continuously capture frames from the camera.
   - The frame is converted to grayscale for better processing.
   - The code detects faces in the frame using the 'face' classifier.

8. **Eye Detection:**
   - For each detected face, the code proceeds to detect left and right eyes.
   - For each eye, it crops the region of interest, converts it to grayscale, and resizes it to 24x24 pixels.
   - The resized eye image is preprocessed and fed into the loaded deep learning model for classification.
   - Predictions are made for both eyes.

9. **Drowsiness Detection:**
   - If both eyes are predicted as 'Closed', the 'score' is incremented, and 'Closed' is displayed on the frame.
   - If any eye is predicted as 'Open', the 'score' is decremented, and 'Open' is displayed on the frame.

10. **Alarm Trigger:**
    - If the 'score' exceeds 15, it means the person is getting drowsy.
    - An alarm sound is played using Pygame.
    - A red rectangle is drawn around the frame to alert the person.

11. **Display and Quit:**
    - The frame with detected eyes and the drowsiness score is displayed.
    - The loop continues until the user presses 'q' in the frame window.
    - After quitting the loop, the video capture is released, and all OpenCV windows are closed.

In summary, this code captures video from the camera, detects faces and eyes, classifies the eye state as open or closed, and triggers an alarm if the person's eyes remain closed for an extended period, indicating drowsiness.

********stepwise explanation of the code in model.py file********

This code is for building, training, and saving a Convolutional Neural Network (CNN) model for image classification using the Keras library. The model is designed to classify images into two categories (hence, the last dense layer has 2 output units with softmax activation). Let's break down the code step by step:

1. **Imports:**
   - `os`: Module for interacting with the operating system, used for file paths.
   - `keras.preprocessing.image`: Keras's image preprocessing module.
   - `matplotlib.pyplot`: Used for plotting.
   - `numpy as np`: NumPy for numerical operations.
   - `keras.utils.np_utils.to_categorical`: Used to one-hot encode the labels.
   - `random` and `shutil`: Used for random operations and file operations.
   - `keras.models.Sequential`: Keras's Sequential model for building neural networks.
   - `keras.layers`: Various layers used in the neural network.
   - `keras.models.load_model`: Used for loading pre-trained models.

2. **Data Generator Function (`generator`):**
   - This function generates a data generator object using Keras's `ImageDataGenerator`.
   - It's used to load and preprocess image data for training and validation.
   - Parameters:
     - `dir`: Directory where the image data is located.
     - `gen`: An `ImageDataGenerator` object for data augmentation and preprocessing.
     - `shuffle`: Boolean to shuffle the data.
     - `batch_size`: Batch size for training.
     - `target_size`: Target size for resizing the input images.
     - `class_mode`: Type of classification problem (in this case, 'categorical' for two classes).

3. **Data Preparation:**
   - `BS` is set to 32, indicating the batch size.
   - `TS` is set to (24, 24), indicating the target image size.
   - `train_batch` and `valid_batch` are created using the `generator` function to load and preprocess training and validation data.

4. **Model Creation:**
   - A Sequential model is created.
   - The model consists of Convolutional, MaxPooling, Dropout, Flatten, and Dense layers.
   - Convolutional layers use 32 and 64 filters, each with a size of (3, 3), and ReLU activation.
   - MaxPooling layers downsample the feature maps.
   - Two Dropout layers are included for regularization.
   - A Flatten layer converts the 2D feature maps to a 1D vector.
   - Two Dense layers are used for classification, with ReLU and softmax activations.

5. **Model Compilation:**
   - The model is compiled with the Adam optimizer and categorical cross-entropy loss.
   - The 'accuracy' metric is used to monitor model performance.

6. **Model Training:**
   - The `fit_generator` method is used to train the model.
   - `train_batch` and `valid_batch` are used as training and validation data sources.
   - The training process runs for 15 epochs (`epochs=15`), and the steps per epoch and validation steps are set based on the data batch sizes.
   
7. **Model Saving:**
   - After training, the model is saved to a file named 'cnnCat2.h5'.

In summary, this code prepares and augments image data, builds a simple CNN model, trains the model using the data, and saves the trained model for future use in image classification tasks.


