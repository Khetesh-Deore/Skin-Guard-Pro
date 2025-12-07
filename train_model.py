import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import os
import numpy as np

# Paths (update if your dataset is elsewhere)
metadata_path = 'HAM10000/ham10000_metadata.csv'
images_dir = 'HAM10000/all_images'  # Create this folder and move all images from part1/part2 into it

# Prepare data (move images to 'all_images' first if not done)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    # Code to copy images from part1/part2 to all_images (run once)
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        for img in os.listdir(os.path.join('HAM10000', part)):
            os.rename(os.path.join('HAM10000', part, img), os.path.join(images_dir, img))

# Load metadata
df = pd.read_csv(metadata_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, x + '.jpg'))
df['dx'] = pd.Categorical(df['dx'])  # Classes: akiec, bcc, bkl, df, mel, nv, vasc

# Split data (80% train, 20% validation)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'])

# Data generators (augmentation for better model)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='image_path', y_col='dx', target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='image_path', y_col='dx', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Build model using transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)  # 7 classes
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers, train top
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_generator, epochs=30, validation_data=val_generator)  # Increase epochs for better accuracy (e.g., 20)

# Save model
model.save('skin_model.h5')
print("Model saved as skin_model.h5")