import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow import keras

# Assuming you have prepared your input features (X) and target labels (y)

# Define your feed-forward deep network architecture
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of folds for cross-validation
k = 5

# Initialize an array to store the accuracy scores for each fold
accuracy_scores = []

# Perform K-fold cross-validation
kf = KFold(n_splits=k, shuffle=True)

for train_index, val_index in kf.split(X):
    # Split the dataset into training and validation sets for the current fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert target labels to one-hot encoded vectors
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
    y_val_encoded = keras.utils.to_categorical(y_val, num_classes)

    # Train the model on the training set for the current fold
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model on the validation set for the current fold
    y_val_pred = model.predict(X_val)
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)

    # Calculate accuracy score for the current fold
    accuracy = accuracy_score(y_val, y_val_pred_labels)

    # Append the accuracy score to the array
    accuracy_scores.append(accuracy)

# Calculate the average accuracy score across all folds
avg_accuracy = np.mean(accuracy_scores)

print("Average Accuracy:", avg_accuracy)
