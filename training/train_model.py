from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import joblib
import numpy as np

# Load dataset
digits = load_digits()
X = digits.images  # 8x8 images
y = to_categorical(digits.target)  # one-hot encode

# Normalize and reshape
X = X / 16.0  # pixel values range from 0–16
X = X.reshape(-1, 8, 8, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Flatten(input_shape=(8, 8, 1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save("digits_model.h5")  # Save Keras model
joblib.dump(digits.target_names, "class_names.json")  # Save label mapping
