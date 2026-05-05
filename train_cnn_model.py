import tensorflow as tf
from tensorflow.keras import layers, models
import os



def build_model(num_classes):
    """Builds a Convolutional Neural Network (CNN) for image classification."""
    model = models.Sequential([
        # Data Augmentation could be added here
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    print("Setting up the ML Model Training Pipeline...")
    
    # Define parameters
    img_height = 224
    img_width = 224
    batch_size = 32
    
    
    dataset_dir = 'dataset' 
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found.")
        print("Please download the dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download")
        print("Extract it and place the 'NORMAL' and 'PNEUMONIA' folders inside your 'dataset' folder.")
        return

    # Load training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Load validation data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Build and train
    model = build_model(num_classes=len(class_names))
    print(model.summary())

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Evaluate the model to clearly show f
    print("\n--- Final Model Evaluation ---")
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Final Testing Accuracy: {val_accuracy * 100:.2f}%\n")

    # Save the trained model to a file
    model.save('medical_image_model.keras')
    print("Model trained and saved as 'medical_image_model.keras'!")

if __name__ == '__main__':
    train_model()
