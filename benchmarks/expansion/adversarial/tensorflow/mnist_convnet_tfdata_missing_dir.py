# ============================================================================
#  AutoFL benchmark expansion — ADVERSARIAL MUTANT A4: mnist_convnet_tfdata_missing_dir
#  Parent   : tensorflow/mnist_convnet.py (sha256 438794ffd9e4e0bbc2f4400ad76598ce37c93481446adbdbc3529951fa2f6d67); the parent is a dev-set script,
#             so this mutant leaks nothing about the holdout set and cannot exist in any training corpus.
#  Change   : training arrays replaced by `keras.utils.image_dataset_from_directory('./data/mnist_png/train', ...)` followed by `.map(augment).cache().shuffle().batch().prefetch(AUTOTUNE)`; `del x_train, y_train`; `model.fit(train_ds, ...)`. The directory does not exist at test time
#  Stresses : I4 (the synthetic fallback must reproduce the element structure of a tf.data pipeline built from a missing directory: image float32 [B,28,28,1] (uint8 PNG scaled by the map), one-hot label float32 [B,10]) plus tf.data handling (map/cache/shuffle/batch/prefetch)
#  Correct conversion : build_dataloader returns a synthetic dataset (tf.data.Dataset or equivalent iterator) with the same element structure when the directory is absent and allow_synthetic_data is set; the model still trains for one step
#  Plausible-but-wrong: silently falling back to `keras.datasets.mnist.load_data()` arrays (which the script no longer trains on), or emitting a NumPy tuple that ignores the pipeline's element structure
#  Notes    : the test arrays (x_test, y_test) are kept in memory for validation/evaluation so that the diff stays within 15 lines; only the TRAINING data goes through the missing-directory tf.data pipeline
#  Diff vs parent: 10 changed lines (+9 / -1), unified diff with zero context below;
#  the header block you are reading is NOT part of the diff (the body after it is parent + diff).
#  --- diff ---
#  --- tensorflow/mnist_convnet.py
#  +++ benchmarks/expansion/adversarial/tensorflow/mnist_convnet_tfdata_missing_dir.py
#  @@ -16,0 +17 @@
#  +import tensorflow as tf
#  @@ -42,0 +44,7 @@
#  +
#  +# Training data now comes from a PNG folder tree (data/mnist_png/train/<label>/*.png) through a tf.data
#  +# pipeline; that directory does NOT exist at test time. The in-memory training arrays are discarded.
#  +train_ds = keras.utils.image_dataset_from_directory("./data/mnist_png/train", color_mode="grayscale", image_size=(28, 28), label_mode="categorical", batch_size=None)
#  +augment = lambda image, label: (tf.cast(image, "float32") / 255.0, label)
#  +train_ds = train_ds.map(augment).cache().shuffle(1024).batch(128).prefetch(tf.data.AUTOTUNE)
#  +del x_train, y_train
#  @@ -72 +80 @@
#  -model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#  +model.fit(train_ds, epochs=epochs, validation_data=(x_test, y_test))
#  --- end diff ---
# ============================================================================
"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
Accelerator: GPU
"""

"""
## Setup
"""

import numpy as np
import keras
from keras import layers
import tensorflow as tf

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Training data now comes from a PNG folder tree (data/mnist_png/train/<label>/*.png) through a tf.data
# pipeline; that directory does NOT exist at test time. The in-memory training arrays are discarded.
train_ds = keras.utils.image_dataset_from_directory("./data/mnist_png/train", color_mode="grayscale", image_size=(28, 28), label_mode="categorical", batch_size=None)
augment = lambda image, label: (tf.cast(image, "float32") / 255.0, label)
train_ds = train_ds.map(augment).cache().shuffle(1024).batch(128).prefetch(tf.data.AUTOTUNE)
del x_train, y_train

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_ds, epochs=epochs, validation_data=(x_test, y_test))

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 8: Image classification](https://deeplearningwithpython.io/chapters/chapter08_image-classification)
"""
