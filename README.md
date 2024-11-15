# Image-Classification-and-Object-Localization

### Project Overview

In this project, we build a Convolutional Neural Network (CNN) from scratch to:
    1. **Classify** the primary subject in an image.
    2. **Localize** the subject by drawing bounding boxes around it.

We'll use the **MNIST dataset** to create a custom dataset where each digit is placed on a 75x75 black canvas at random locations. We also calculate bounding boxes for each digit, and the model will predict these bounding boxes as a **regression task** (outputting numeric values instead of class labels).

### Key Features
- **Data Preprocessing**: Each digit image is resized and randomly positioned on a larger canvas.
- **Model Training**: The CNN model is trained to both classify the digit and predict its bounding box.
- **Bounding Box Prediction**: Uses regression techniques to determine coordinates for bounding boxes.
- **Visualization**: Helper functions draw bounding boxes on images for easy verification of predictions.

### Visualizing Results
The project includes utilities to visualize bounding boxes and check model performance visually.

![]("https://github.com/ParitKansal/photos/blob/main/download%20(1).png")

!![fcfeccrcrrvrc]("https://github.com/ParitKansal/photos/blob/main/download%20(2).png")

!![fcfeccrcrrvrc]("https://github.com/ParitKansal/photos/blob/main/download%20(3).png")
