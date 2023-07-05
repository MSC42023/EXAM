import numpy as np
from tensorflow import keras

# Assuming you have prepared your input features (X) and target labels (y)

# Placeholder values for X and y
X = np.random.rand(1000, 10)  # Assuming 1000 samples with 10 input features
y = np.random.randint(0, 5, size=(1000,))  # Assuming 5 classes and 1000 target labels

# Normalize input features between 0 and 1
X = X / np.max(X)

# Convert target labels to one-hot encoded vectors
num_classes = len(np.unique(y))
y_encoded = keras.utils.to_categorical(y, num_classes)

# Define your feed-forward deep network architecture
input_shape = X.shape[1]  # Number of input features
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on your dataset
model.fit(X, y_encoded, epochs=10, batch_size=32)

# Make predictions on new data
new_data = np.random.rand(5, 10)  # Assuming 5 samples of new data with 10 input features
new_data_normalized = new_data / np.max(new_data)
predictions = model.predict(new_data_normalized)

# Get class probabilities for each prediction
class_probabilities = predictions.tolist()

# Print class probabilities for each prediction
for i, probs in enumerate(class_probabilities):
    print(f"Prediction {i+1} - Class Probabilities:", probs)
