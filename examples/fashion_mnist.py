"""Creates the same model as in
https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a

In that article:

.. code:: python

    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # Take a look at the model summary
    model.summary()
"""
from torchluent import FluentModule

def model():
    return (
        FluentModule((1, 28, 28))  # features, height, width
        .verbose()
        .conv2d(64, 2, padding=1)
        .operator('ReLU')
        .maxpool2d(2)
        .operator('Dropout', 0.3)
        .conv2d(32, 2, padding=1)
        .operator('ReLU')
        .flatten()
        .dense(256)
        .operator('ReLU')
        .operator('Dropout', 0.5)
        .dense(10)
        .build()
    )

if __name__ == '__main__':
    print(model())
