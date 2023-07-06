import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense

# Generate sample data for multiclass classification
# Replace this with your own dataset or data generation logic
num_samples = 1000
input_dim = 10
num_classes = 3

# Generate random input data
X = np.random.random((num_samples, input_dim))
# Generate random target labels for multiclass classification
y = np.random.randint(num_classes, size=(num_samples,))

# Convert output labels to one-hot encoded vectors
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Define the number of folds for cross-validation
num_folds = 5

# Initialize the K-Fold object
kf = KFold(n_splits=num_folds, shuffle=True)

# Create lists to store the evaluation metrics for each fold
accuracy_scores = []

# Iterate over each fold
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create the model
    model = Sequential()
    
    # Add the first hidden layer with 64 units and 'relu' activation
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    
    # Add the second hidden layer with 32 units and 'relu' activation
    model.add(Dense(32, activation='relu'))
    
    # Add the output layer with 'softmax' activation for multiclass classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model with appropriate loss and optimizer for multiclass classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Evaluate the model on the test set
    _, accuracy_score = model.evaluate(X_test, y_test)
    
    # Store the evaluation score
    accuracy_scores.append(accuracy_score)

# Calculate the average accuracy score across all folds
average_accuracy = np.mean(accuracy_scores)

# Print the average accuracy score
print("Average Accuracy:", average_accuracy)
