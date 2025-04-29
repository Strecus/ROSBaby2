import pandas as pd
import numpy as np
import pickle
import os
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow imports for LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set paths based on operating system
system = platform.system()
if system == 'Darwin':  # macOS
    landmarks_file = 'training/pose_landmarks_dataset.npz'
    model_output_dir = 'training/models'
else:  # Windows or other OS
    landmarks_file = 'pose_landmarks_dataset.npz'
    model_output_dir = 'models'

print(f"Operating system detected: {system}")
print(f"Using landmarks file: {landmarks_file}")
print(f"Using model output directory: {model_output_dir}")

# Create output directory
os.makedirs(model_output_dir, exist_ok=True)

print("Loading landmark data...")
# Load the landmark data
try:
    data = np.load(landmarks_file, allow_pickle=True)
    landmarks_data = data['data']
    print(f"Loaded data with {len(landmarks_data)} samples")
except FileNotFoundError:
    print(f"Error: {landmarks_file} not found. Run pose_landmark_extraction.py first.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Prepare data for LSTM (reshape to 3D: [samples, time_steps, features])
X = []
y = []
sequence_length = 10  # The number of frames in each sequence

# Extract sequences and labels
for item in landmarks_data:
    label = item['label']
    landmarks = item['landmarks']
    
    # We need to reshape from (sequence_length, 33*4) to (sequence_length, 132)
    # where 132 is 33 landmarks * 4 values per landmark
    reshaped_seq = np.array([frame.reshape(-1) if len(frame.shape) > 1 else frame 
                             for frame in landmarks])
    
    X.append(reshaped_seq)
    y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Print distribution of classes
print("Class distribution:")
unique_labels, counts = np.unique(y, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count} samples")

# Create a label encoder to convert text labels to numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to categorical for Keras
num_classes = len(unique_labels)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y_categorical, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features (normalize each feature independently)
# We need to reshape to 2D for scaling, then back to 3D
n_samples_train = X_train.shape[0]
n_samples_test = X_test.shape[0]
n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

# Reshape to 2D
X_train_reshaped = X_train.reshape(n_samples_train * n_timesteps, n_features)
X_test_reshaped = X_test.reshape(n_samples_test * n_timesteps, n_features)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back to 3D
X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

print(f"\nTraining LSTM model with data shape: {X_train_scaled.shape}")

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
print("\nTraining LSTM model...")
history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model performance...")
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Get predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)

# Convert numeric predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_test_decoded)

# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Create and save confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(model_output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
print(f"Saved confusion matrix to {cm_path}")

# Save the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()

history_path = os.path.join(model_output_dir, 'training_history.png')
plt.savefig(history_path)
print(f"Saved training history to {history_path}")

# Save the TensorFlow model
model_save_path = os.path.join(model_output_dir, 'lstm_model.keras')
model.save(model_save_path)

# Save the scaler and label encoder using pickle
scaler_path = os.path.join(model_output_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

encoder_path = os.path.join(model_output_dir, 'label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model saved to {model_save_path}")
print(f"Scaler saved to {scaler_path}")
print(f"Label encoder saved to {encoder_path}")
print("\nTraining complete!") 