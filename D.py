import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configure GPU settings for RTX 4050
print("="*60)
print("GPU Configuration")
print("="*60)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"✓ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        # Set GPU as default device
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✓ {len(logical_gpus)} Logical GPU(s) available")
        print("✓ GPU memory growth enabled")
        print("✓ Training will use GPU acceleration")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU found. Training will use CPU (slower)")
    print("  Make sure you have installed tensorflow-gpu or tensorflow with CUDA support")

print("="*60 + "\n")

class ASLRecognitionSystem:
    def __init__(self, dataset_path, img_size=(64, 64)):
        """
        Initialize the ASL Recognition System
        
        Args:
            dataset_path: Path to the ASL alphabet dataset
            img_size: Target image size (width, height)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = []
        
    def load_and_preprocess_data(self):
        """Load ALL images and labels from the dataset"""
        print(f"Loading complete dataset (ALL available images)...")
        
        images = []
        labels = []
        
        # The dataset structure
        train_path = os.path.join(self.dataset_path, 'asl_alphabet_train')
        
        if not os.path.exists(train_path):
            train_path = self.dataset_path
        
        # Get all class folders
        class_folders = sorted([f for f in os.listdir(train_path) 
                               if os.path.isdir(os.path.join(train_path, f))])
        
        self.class_names = class_folders
        print(f"Found {len(self.class_names)} classes: {self.class_names}\n")
        
        # Load ALL images from each class
        total_images = 0
        for idx, class_name in enumerate(class_folders):
            class_path = os.path.join(train_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading class '{class_name}': {len(image_files)} images", end='')
            
            loaded_count = 0
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        
                        images.append(img)
                        labels.append(idx)
                        loaded_count += 1
                        
                        # Show progress every 500 images
                        if loaded_count % 500 == 0:
                            print(f".", end='', flush=True)
                except Exception as e:
                    continue
            
            total_images += loaded_count
            print(f" ✓ ({loaded_count} loaded, Total: {total_images})")
        
        # Convert to numpy arrays
        print("\nConverting to arrays...")
        X = np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]
        y = np.array(labels)
        
        print(f"\n{'='*60}")
        print(f"Dataset loaded successfully!")
        print(f"{'='*60}")
        print(f"Total images: {X.shape[0]}")
        print(f"Image shape: {X.shape[1:]}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Memory usage: ~{X.nbytes / (1024**3):.2f} GB")
        print(f"{'='*60}\n")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=42, stratify=y_train_val
        )
        
        print(f"Training set: {X_train.shape[0]} images ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} images ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} images ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
        print()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip for sign language
            fill_mode='nearest'
        )
    
    def build_model(self, num_classes):
        """Build CNN model for sign language recognition"""
        print("Building model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train the model with callbacks"""
        print("Starting training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}\n")
        
        # Create data augmentation
        datagen = self.create_data_augmentation()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_asl_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model on test set...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.class_names))
        
        return y_pred, test_acc
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved as 'training_history.png'")
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - ASL Recognition', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    def save_model(self, filepath='asl_model_final.keras'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"\n✓ Model saved as '{filepath}'")
        
        # Save class names
        np.save('class_names.npy', self.class_names)
        print("✓ Class names saved as 'class_names.npy'")
    
    def load_model(self, filepath='asl_model_final.keras'):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        self.class_names = np.load('class_names.npy', allow_pickle=True).tolist()
        print(f"✓ Model loaded from '{filepath}'")
        return self.model
    
    def predict_image(self, img_path):
        """Predict sign language letter from an image"""
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.img_size)
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Display result
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {self.class_names[predicted_class]} "
                 f"(Confidence: {confidence:.2%})", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return self.class_names[predicted_class], confidence


# Main execution
if __name__ == "__main__":
    # Initialize the system - USE ALL IMAGES
    DATASET_PATH = r"C:\Users\acer\Downloads\archive (1)\asl_alphabet_train"
    
    print("="*60)
    print("ASL Alphabet Recognition System - FULL DATASET")
    print("="*60)
    print("This will load ALL available images from the dataset")
    print("="*60 + "\n")
    
    asl_system = ASLRecognitionSystem(DATASET_PATH, img_size=(64, 64))
    
    # Load and preprocess ALL data
    X, y = asl_system.load_and_preprocess_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = asl_system.split_data(X, y)
    
    # Build model
    num_classes = len(asl_system.class_names)
    model = asl_system.build_model(num_classes)
    
    # Train model
    history = asl_system.train_model(X_train, y_train, X_val, y_val, 
                                     epochs=50, batch_size=64)
    
    # Evaluate model
    y_pred, test_acc = asl_system.evaluate_model(X_test, y_test)
    
    # Plot results
    asl_system.plot_training_history()
    asl_system.plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    asl_system.save_model('asl_model_final.keras')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Total images used: {len(X)}")
    print(f"Classes: {len(asl_system.class_names)}")
    print("\nModel saved and ready for deployment!")
    print("="*60)
    
    # Example: Load and use the model later
    # asl_system.load_model('asl_model_final.keras')
    # prediction, confidence = asl_system.predict_image('path/to/test/image.jpg')