# SC Lab - Experiment 10: CNN classifier.
# A convolutional neural network that classifies handwritten digits (MNIST),
# built with TensorFlow/Keras. Install the dependencies first:
#     pip install -r requirements.txt

import tensorflow as tf
from tensorflow.keras import layers, models


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalise to [0, 1] and add the single (grayscale) channel dimension.
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.summary()

    model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_split=0.1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f}")
