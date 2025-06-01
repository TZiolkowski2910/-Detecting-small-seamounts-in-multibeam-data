import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K

# Register custom functions to allow proper model loading
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(K.cast(y_true, dtype='float32'))  # Convert to float32
    y_pred_f = K.flatten(y_pred)  # y_pred is already float32
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def mean_iou(y_true, y_pred):
    metric = MeanIoU(num_classes=2)
    return metric(y_true, y_pred)

# Directories
model_path = r""
input_dir = r""
output_dir = r""

# Parameters
image_size = (256, 256)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load the model with properly registered functions
print("Loading model...")
with tf.keras.utils.custom_object_scope({"dice_loss": dice_loss, "dice_coefficient": dice_coefficient, "mean_iou": mean_iou, "MeanIoU": MeanIoU}):
    model = load_model(model_path)

# Function to load and prepare images
def load_and_prepare_image(image_path, size):
    img = Image.open(image_path).convert("RGB").resize(size)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return img_array

# Process images
print("Generating masks for images...")
for file in sorted(os.listdir(input_dir)):
    if file.endswith(".png"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"combined_{file}")

        # Load and prepare the image
        input_image = load_and_prepare_image(input_path, image_size)
        input_image_batch = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Predict mask
        predicted_mask = model.predict(input_image_batch)[0]
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Apply threshold

        # Create figure without labels, titles, or axes
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(input_image)
        ax[1].imshow(predicted_mask[:, :, 0], cmap="gray")

        # Remove axes
        for a in ax:
            a.axis("off")

        # Save the combined image
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        print(f"Saved: {output_path}")

print("Seamounts detected.")
