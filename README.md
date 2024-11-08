# sleep-detection
# aim to develop a machine learning based solution that detects and analyze the sleep.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
mport tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from flask import Flask, request, jsonify
# Generate synthetic sleep data (or replace with real data from wearables)
np.random.seed(42)
data = pd.DataFrame({
    'heart_rate': np.random.normal(70, 10, 1000),
    'movement': np.random.normal(0.5, 0.2, 1000),
    'eye_movement': np.random.choice([0, 1], 1000),  # 0 for still, 1 for movement
    'sleep_stage': np.random.choice(['Awake', 'Light', 'Deep', 'REM'], 1000)
})
# Map sleep stages to numerical labels
data['sleep_stage'] = data['sleep_stage'].map({'Awake': 0, 'Light': 1, 'Deep': 2, 'REM': 3})
print(data.head())
# Features and target variable and using train-test 
X = data[['heart_rate', 'movement', 'eye_movement']]
y = data['sleep_stage']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# from tensorflow and its methods
# Build the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes for 4 sleep stages
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")
# Predict sleep stages
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
# Compare predictions with actual values
print("Predicted sleep stages:", predicted_classes[:10])
print("Actual sleep stages:", y_test[:10].values)
# using matplotlib
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# from flask
app = Flask(_name_)
@app.route('/predict', methods=['POST'])
def predict():
# Get JSON data from the request
    data = request.get_json()
    features = np.array([data['heart_rate'], data['movement'], data['eye_movement']]).reshape(1, -1)
   # Predict using the trained model
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
     return jsonify({'predicted_sleep_stage': int(predicted_class)})
i __name__ == '__main__':
    app.run(debug=True)
