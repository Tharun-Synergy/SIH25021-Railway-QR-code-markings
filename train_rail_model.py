import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Data paths
train_dir = 'QR_SCANNER/dataset_rail/train'
val_dir = 'QR_SCANNER/dataset_rail/val'

# Check if directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise ValueError(f"Dataset not found! Run generate_rail_dataset.py first. Train: {train_dir}, Val: {val_dir}")

# Heavy augmentation for rail conditions (grayscale)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Angles from rail views
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,  # Perspective distortion
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  # Lighting variations
    channel_shift_range=20.0,  # Color shifts for rust (works on grayscale too)
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Generators (auto-maps classes alphabetically: 'no_qr':0, 'qr':1)
train_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(224, 224), 
    color_mode='grayscale', 
    batch_size=32, 
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, 
    target_size=(224, 224),
    color_mode='grayscale', 
    batch_size=32, 
    class_mode='binary'
)

# Print class indices to confirm (QR should be 1)
print("Class indices:", train_gen.class_indices)  # Expected: {'no_qr': 0, 'qr': 1}

# Custom CNN Model (Grayscale-friendly)
model = models.Sequential([
    # Input: Grayscale (1 channel)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Extra layer for rust textures
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting on augmented data
    layers.Dense(1, activation='sigmoid')  # Binary: QR present (1) or not (0)
])

# Compile with balanced metrics (precision/recall for detection)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Standard LR for custom CNN
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Summary (optional: prints model architecture)
model.summary()

# Train with callbacks (FIXED: .keras extension)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('rail_qr_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen, 
    epochs=20, 
    validation_data=val_gen, 
    callbacks=callbacks,
    verbose=1  # Show progress per epoch
)

# Save final model (FIXED: .keras extension)
model.save('rail_qr_cnn_model.keras')
print('Railtrack QR model trained and saved to rail_qr_cnn_model.keras!')

# Print final metrics
best_val_acc = max(history.history['val_accuracy'])
print(f"Best Val Accuracy: {best_val_acc:.2f}")
print(f"Final Val Precision: {history.history['val_precision'][-1]:.2f}")
print(f"Final Val Recall: {history.history['val_recall'][-1]:.2f}")

# Tip: High recall (>0.9) is key for detecting rusty QRs; precision avoids false alarms