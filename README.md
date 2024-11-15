# **Image Classification and Object Localization**

The **Image Classification and Object Localization** project demonstrates how to build a Convolutional Neural Network (CNN) from scratch to perform both classification and localization of the main subject in an image, using the MNIST dataset as the base.

### Project Steps:

1. **Dataset Preparation**:
   - Each MNIST digit image is resized and placed randomly on a 75x75 black canvas to simulate a "localization" task.
   - Bounding box coordinates are generated for each image, which the model will learn to predict.

2. **Model Design**:
   - The CNN is configured to handle two outputs:
     - **Classification** of the digit (e.g., 0-9).
     - **Bounding box regression** to predict the bounding box coordinates.

3. **Model Training**:
   - The model is trained to minimize both classification and bounding box prediction errors. It uses mean squared error (MSE) as the loss metric for bounding box regression.

4. **Visualization**:
   - The project includes utilities for drawing bounding boxes on images to visually verify model predictions.

### Key Visualizations:

- **Bounding Box MSE VS Epochs**
  
  ![](https://github.com/ParitKansal/photos/blob/main/download%20(1).png)

- **Classification Loss MSE VS Epochs**
  
  !![](https://github.com/ParitKansal/photos/blob/main/download%20(3).png)

- **Classification Accuracy MSE VS Epochs**

  !![](https://github.com/ParitKansal/photos/blob/main/download%20(2).png)

### Model and Predictions

#### Model Download:
- Download the trained model [here](https://github.com/ParitKansal/Image-Classification-and-Object-Localization/blob/main/model.h5).

#### Predicting with the Model:

To use the model, upload an image to Google Colab and load the model. Here’s a complete script to help you test the model and visualize bounding boxes on a new digit image.

```python
from google.colab import files  # Only works in Google Colab
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from IPython.display import Markdown, display

im_width = 75
im_height = 75
use_normalized_coordinates = True

def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=1, display_str_list=()):
    """Draws bounding boxes on image (numpy array)."""
    image_pil = Image.fromarray(image)
    rgbimg = Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)
    draw_bounding_boxes_on_image(rgbimg, boxes, color, thickness, display_str_list)
    return np.array(rgbimg)

def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=1, display_str_list=()):
    """Draws bounding boxes on image."""
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3], boxes[i, 2], color[i], thickness, display_str_list[i])

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=1, display_str=None, use_normalized_coordinates=True):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

def display_single_digit_with_predicted_bboxes(validation_digit, predicted_bboxes):
    """
    Display a single digit image with its predicted bounding boxes.
    """
    validation_digit = validation_digit * 255.0
    validation_digit = validation_digit.reshape(75, 75)

    fig, ax = plt.subplots(figsize=(6, 6))
    img_to_draw = draw_bounding_boxes_on_image_array(
        image=validation_digit,
        boxes=predicted_bboxes,
        color=['red'],
        display_str_list=["pred"]
    )

    ax.imshow(img_to_draw)
    ax.axis('off')
    plt.show()

def upload_image():
    uploaded = files.upload()  # Trigger the file upload dialog
    for filename in uploaded.keys():
        img = Image.open(BytesIO(uploaded[filename])).convert('L')
        img = img.resize((75, 75))
        img_array = np.array(img) / 255.0
        return img_array

def upload_model():
    uploaded = files.upload()  # Trigger the file upload dialog for the model
    for filename in uploaded.keys():
        model_file_path = f"/content/{filename}"
        with open(model_file_path, "wb") as f:
            f.write(uploaded[filename])
    return model_file_path

# Upload the model manually
display(Markdown("# UPLOAD MODEL : "))
model_file_path = upload_model()
display(Markdown("---"))

# Load the trained model from the local file
model = load_model(model_file_path)

# Example usage (make sure to replace with actual data):
display(Markdown("# UPLOAD Image : "))
validation_digit = upload_image()
display(Markdown("---"))

# Reshape the image to match the expected input shape (1, 75, 75, 1)
validation_digit = validation_digit.reshape(1, 75, 75, 1)

# Model prediction
predictions = model.predict(validation_digit)

# Extract predicted labels (class predictions)
predicted_labels = np.argmax(predictions[0], axis=1)

# Extract predicted bounding boxes (assuming it's the second output of the model)
predicted_bboxes = predictions[1]  # Should be of shape [num_boxes, 4]

# Display the result
display(Markdown(f"---"))
display(Markdown(f"## **Digit Label:** {predicted_labels[0]}"))
display_single_digit_with_predicted_bboxes(validation_digit[0], predicted_bboxes)
```

### Instructions:

1. **Upload the model file** when prompted.
2. **Upload an image file** of a digit to test classification and bounding box localization.
3. Run the code to visualize the digit with its predicted bounding box.

This project showcases both image classification and bounding box regression using CNN, handling MNIST digits in a localization task, an effective blend of classification and localization.
