import numpy as np
import tensorflow as tf
from tensorflow import keras

# Prepare the dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Define the model using Keras Sequential API
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Input layer with 8 neurons and ReLU activation
    keras.layers.Dense(1, activation='sigmoid')   # Output layer with 1 neuron and Sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(X)
print('Predictions:', np.round(predictions).flatten())

output:
1/1 [==============================] - 0s 230ms/step - loss: 0.1760 - accuracy: 1.0000
Loss: 0.1760, Accuracy: 1.0000
1/1 [==============================] - 0s 138ms/step
Predictions: [0. 1. 1. 0.]
