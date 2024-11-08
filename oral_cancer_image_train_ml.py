import pickle
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Paths to the dataset folders (Ensure these paths are correct)
train_path = "/home/koosha/Downloads/AIDATA/oral/Katebi/oral_lesions/train"
val_path = "/home/koosha/Downloads/AIDATA/oral/Katebi/oral_lesions/val"

# Image dimensions (resize images to this shape)
IMG_SIZE = (64, 64)  # Modify based on memory limitations or model requirement

# Function to load images and labels
def load_images_labels(folder_path):
    images, labels = [], []
    for label_name in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label_name)
        if os.path.isdir(label_folder):
            for file_name in os.listdir(label_folder):
                if file_name.endswith('.jpg'):
                    file_path = os.path.join(label_folder, file_name)
                    try:
                        image = Image.open(file_path).convert('RGB')
                        image = image.resize(IMG_SIZE)
                        images.append(np.array(image).flatten())  # Flatten to 1D
                        labels.append(label_name)
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
    return images, labels

# Load train and validation data
X_train, y_train = load_images_labels(train_path)

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(y_train)  # Fit on training labels
y_train_encoded = label_encoder.transform(y_train)

# Build the model
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train_encoded)

# Make predictions on the training data
y_train_pred = model.predict(X_train)

# Calculate accuracy on training data
train_accuracy = accuracy_score(y_train_encoded, y_train_pred)

# Save the trained model to a pickle file
model_filename = "oral_cancer_model_ml.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
    print("model Saved with Name : oral_cancer_model_ml.pkl")

# Output accuracy of the training data
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
