
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CrackDetectionCNN:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def create_model(self):
        """
        Create CNN model for crack detection
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer (binary classification: crack/no crack)
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
    def prepare_data(self, data_dir):
        """
        Prepare training and validation datasets
        """
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """
        Train the model
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_crack_detection_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_generator):
        """
        Evaluate model performance
        """
        # Predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_generator.classes
        
        # Classification report
        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=['No Crack', 'Crack']))
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Crack', 'Crack'],
                   yticklabels=['No Crack', 'Crack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return predictions, predicted_classes
    
    def predict_single_image(self, image_path):
        """
        Predict crack for a single image
        """
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        img_array = img_array / 255.0  # Normalize
        
        prediction = self.model.predict(img_array)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            result = f"Crack detected with {confidence:.2%} confidence"
        else:
            result = f"No crack detected with {(1-confidence):.2%} confidence"
        
        return result, confidence

def main():
    # Initialize the crack detection system
    crack​​​​​​​​​​​​​​​​
