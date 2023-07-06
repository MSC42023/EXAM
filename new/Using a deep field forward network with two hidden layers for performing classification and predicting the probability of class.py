import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate sample data for classification
# Replace this with your own dataset or data generation logic
num_samples = 1000
input_dim = 10
output_dim = 2

# Generate random input data
X = np.random.random((num_samples, input_dim))
# Generate random output labels (classes)
y = np.random.randint(output_dim, size=(num_samples,))

# Create the model
model = Sequential()

# Add the first hidden layer with 64 units and 'relu' activation
model.add(Dense(64, activation='relu', input_dim=input_dim))

# Add the second hidden layer with 32 units and 'relu' activation
model.add(Dense(32, activation='relu'))

# Add the output layer with 'softmax' activation for classification
model.add(Dense(output_dim, activation='softmax'))

# Compile the model with appropriate loss and optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions on new data
# Replace X_test with your own test data
X_test = np.random.random((10, input_dim))
predictions = model.predict(X_test)

# Print the predicted probabilities for each class
print("Predicted probabilities:")
for pred in predictions:
    print(pred)
