import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate sample data for linear regression
# Replace this with your own dataset or data generation logic
num_samples = 1000
input_dim = 10

# Generate random input data
X = np.random.random((num_samples, input_dim))
# Generate random target values for linear regression
y = np.random.random((num_samples, 1))

# Create the model
model = Sequential()

# Add the first hidden layer with 64 units and 'relu' activation
model.add(Dense(64, activation='relu', input_dim=input_dim))

# Add the second hidden layer with 32 units and 'relu' activation
model.add(Dense(32, activation='relu'))

# Add the output layer with one unit for linear regression
model.add(Dense(1))

# Compile the model with appropriate loss and optimizer for linear regression
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions on new data
# Replace X_test with your own test data
X_test = np.random.random((10, input_dim))
predictions = model.predict(X_test)

# Print the predicted values
print("Predicted values:")
for pred in predictions:
    print(pred[0])
