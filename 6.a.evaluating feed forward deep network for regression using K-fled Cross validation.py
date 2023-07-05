import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

# Assuming you have prepared your input features (X) and target values (y)

# Define your feed-forward deep network architecture
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))  # Assuming single output for regression

# Compile the model with an appropriate loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Define the number of folds for cross-validation
k = 5

# Initialize arrays to store the evaluation metrics for each fold
mse_scores = []
mae_scores = []

# Perform K-fold cross-validation
kf = KFold(n_splits=k, shuffle=True)

for train_index, val_index in kf.split(X):
    # Split the dataset into training and validation sets for the current fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train the model on the training set for the current fold
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model on the validation set for the current fold
    y_val_pred = model.predict(X_val)

    # Calculate evaluation metrics for the current fold
    mse = np.mean((y_val_pred - y_val) ** 2)
    mae = np.mean(np.abs(y_val_pred - y_val))

    # Append the evaluation scores to the arrays
    mse_scores.append(mse)
    mae_scores.append(mae)

# Calculate the average evaluation metrics across all folds
avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)

print("Average MSE:", avg_mse)
print("Average MAE:", avg_mae)
