import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense

# Generate sample data for regression
# Replace this with your own dataset or data generation logic
num_samples = 1000
input_dim = 10

# Generate random input data
X = np.random.random((num_samples, input_dim))
# Generate random target values for regression
y = np.random.random((num_samples, 1))

# Define the number of folds for cross-validation
num_folds = 5

# Initialize the K-Fold object
kf = KFold(n_splits=num_folds, shuffle=True)

# Create lists to store the evaluation metrics for each fold
mse_scores = []
mae_scores = []

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
    
    # Add the output layer with one unit for regression
    model.add(Dense(1))
    
    # Compile the model with appropriate loss and optimizer for regression
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Evaluate the model on the test set
    mse_score = model.evaluate(X_test, y_test)
    mae_score = np.mean(np.abs(model.predict(X_test) - y_test))
    
    # Store the evaluation scores
    mse_scores.append(mse_score)
    mae_scores.append(mae_score)

# Calculate the average evaluation scores across all folds
average_mse = np.mean(mse_scores)
average_mae = np.mean(mae_scores)

# Print the average evaluation scores
print("Average MSE:", average_mse)
print("Average MAE:", average_mae)
