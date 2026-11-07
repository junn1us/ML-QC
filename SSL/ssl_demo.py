# Autoencoder SSL mock up
import tensorflow as tf
from temsorflow.keras import models, layer
# this gives the framework for the multilayer perceptron 
import numpy as np
import matplotlib.pyplot as plt

# define train and test
(x_train, _), (x_test, _) = tf.keras.dataset.mnist.load_data()

# define train and test sets, set data type and dimensions
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train_small = x_train[:1000]
x_test_small = x_test[:200]

# set rotation angles
angles = [0, 90, 180, 270]

# define a rotated image
def rotate_images(images, angles):
    rotated_images = []
    labels = []
    for img in images:
        for i, angle in enumerate(angles):
            rotated = tf.image.rot90(img, k=angle // 90)
            rotated_images.append(rotated.numpy())
            labels.append(i)
    return np.array(rotated_images), np.array(labels)

# create new dataset, set task to present the rotated angle
x_train_rot, y_train_rot = rotate_images(x_train_small, angles)
x_test_rot, y_test_rot = rotate_images(x_test_small, angles)

# define and compile the CNN
model = models.Sequential 
layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(angles), activation='softmax')
])
#compile model, select optimizer,loss and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model, upload rotated images
model.fit(x_train_rot, y_train_rot, epochs=5, batch_size=64,
          validation_data=(x_test_rot, y_test_rot))

# visualize the rotated results
predictions = model.predict(x_test_rot)

num_examples = 5
indices = np.random.choice(len(x_test_rot), num_examples, replace=False)

for i, idx in enumerate(indices):
    img = x_test_rot[idx].squeeze()
    true_label = y_test_rot[idx]
    pred_label = np.argmax(predictions[idx])

    plt.subplot(1, num_examples, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {angles[true_label]}°\nPred: {angles[pred_label]}°")
    plt.axis('off')

plt.show()
