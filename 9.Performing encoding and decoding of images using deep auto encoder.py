import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the dataset (e.g., MNIST)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the input images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Define the autoencoder architecture
input_shape = (28, 28, 1)
latent_dim = 16

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_dim)
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    tf.keras.layers.Dense(7 * 7 * 8, activation='relu'),
    tf.keras.layers.Reshape((7, 7, 8)),
    tf.keras.layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),
    tf.keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the autoencoder
autoencoder = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Encode and decode images
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)

# Visualize original and reconstructed images
n_images = 10
plt.figure(figsize=(20, 4))
for i in range(n_images):
    # Display original images
    ax = plt.subplot(2, n_images, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n_images, i + 1 + n_images)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
