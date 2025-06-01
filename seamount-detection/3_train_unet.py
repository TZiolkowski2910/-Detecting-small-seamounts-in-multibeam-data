import os
import numpy as np
import tensorflow as tf
import pandas as pd
import kerastuner as kt
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Directories
input_dir = r""
mask_dir = r""
results_csv = r""

# Parameters
image_size = (256, 256)
epochs = 50
max_trials = 1  # Number of hyperparameter tuning trials

# Functions to load images
def load_images(directory, size, grayscale=False):
    images = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".png"):
            img = Image.open(os.path.join(directory, file))
            img = img.convert("L") if grayscale else img.convert("RGB")
            img = img.resize(size)
            images.append(np.array(img))
    return np.array(images)

# Load data
print("Loading input images...")
inputs = load_images(input_dir, image_size) / 255.0

print("Loading masks...")
masks = load_images(mask_dir, image_size, grayscale=True)
masks = np.expand_dims(masks, axis=-1)  # Convert to (256, 256, 1)
masks = (masks > 0).astype(np.uint8)  # Convert to binary mask

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(inputs, masks, test_size=0.2, random_state=42)

# Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(K.cast(y_true, dtype='float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Hyperparameter tuning function
def build_tunable_unet(hp):
    inputs = Input((image_size[0], image_size[1], 3))

    # Tune number of filters and kernel size
    filters = hp.Choice('filters', [32])
    kernel_size = hp.Choice('kernel_size', [5])
    dropout_rate = hp.Choice('dropout_rate', [0.1]) #  hp.Float('dropout_rate', 0.1, 0.5, step=0.1)

    # Encoder
    c1 = Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same')(inputs)
    c1 = Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same')(c1)
    c1 = Dropout(dropout_rate)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu', padding='same')(p1)
    c2 = Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu', padding='same')(c2)
    c2 = Dropout(dropout_rate)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu', padding='same')(p2)
    c3 = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu', padding='same')(c3)
    c3 = Dropout(dropout_rate)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu', padding='same')(p3)
    c4 = Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu', padding='same')(c4)
    c4 = Dropout(dropout_rate)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu', padding='same')(p4)
    c5 = Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu', padding='same')(u6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu', padding='same')(u7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu', padding='same')(u8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same')(u9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs, outputs)

    # Tune learning rate and batch size
    learning_rate = hp.Choice('learning_rate', [1e-4])
    batch_size = hp.Choice('batch_size', [16])

    # Define MeanIoU as a global variable
    mean_iou_metric = MeanIoU(num_classes=2)

    def mean_iou(y_true, y_pred):
        return mean_iou_metric(y_true, y_pred)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=dice_loss,
        metrics=[dice_coefficient, mean_iou]
    )

    return model

# Define checkpoint to save the best model based on val_loss
model_checkpoint = ModelCheckpoint(
    r"C:\Users\Tobi\Desktop\Seamounts\UNet_Input\Chunk24\unet_model_64_32_5_0.5_1e-4.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)

# Define tuner
tuner = kt.RandomSearch(
    build_tunable_unet,
    objective='val_loss',
    max_trials=max_trials,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='unet_tuning'
)

# Run hyperparameter search
print("Starting hyperparameter tuning...")
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16,
             callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True), model_checkpoint],
             verbose=1)

# Get best parameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

results_data = []

for trial in tuner.oracle.trials.values():
    trial_data = {
        'filters': trial.hyperparameters.values.get('filters', None),
        'kernel_size': trial.hyperparameters.values.get('kernel_size', None),
        'dropout_rate': trial.hyperparameters.values.get('dropout_rate', None),
        'learning_rate': trial.hyperparameters.values.get('learning_rate', None),
        'batch_size': trial.hyperparameters.values.get('batch_size', None),
        'val_loss': trial.metrics.get_best_value('val_loss') if 'val_loss' in trial.metrics.metrics else None,
        'mean_iou': trial.metrics.get_best_value('mean_iou') if 'mean_iou' in trial.metrics.metrics else None,
        'val_mean_iou': trial.metrics.get_best_value('val_mean_iou') if 'val_mean_iou' in trial.metrics.metrics else None
    }
    results_data.append(trial_data)

df = pd.DataFrame(results_data)
df.to_csv(results_csv, index=False)
print(f"Hyperparameter results saved to {results_csv}")

print("Tuning complete.")
