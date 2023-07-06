import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Generate sample data for binary classification
# Replace this with your own dataset or data generation logic
num_samples = 1000
input_dim = 10

# Generate random input data
X = np.random.random((num_samples, input_dim))
# Generate random binary labels
y = np.random.randint(2, size=(num_samples,))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()

# Add the input layer
model.add(Dense(64, activation='relu', input_dim=input_dim))
model.add(Dropout(0.5))  # Dropout layer with 50% dropout rate

# Add a hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer with 50% dropout rate

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model with appropriate loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
