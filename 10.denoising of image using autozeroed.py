import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Assuming you have your noisy image data stored in the 'noisy_image' variable

# Normalize the pixel values between 0 and 1
noisy_image = noisy_image / 255.0

# Define the autoencoder architecture
input_shape = noisy_image.shape
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.UpSampling2D((2, 2)))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.UpSampling2D((2, 2)))
model.add(keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model with appropriate loss function and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder using the noisy image as both input and target
model.fit(noisy_image, noisy_image, epochs=10, batch_size=16)

# Use the trained autoencoder to denoise the image
denoised_image = model.predict(noisy_image)

# Plot the original noisy image and the denoised image
plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.show()
