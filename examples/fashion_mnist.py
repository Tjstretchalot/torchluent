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
import torch.nn as nn

class ModelPureTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = nn.Conv2d(1, 64, 2, padding=1)
        self.relu_1 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(2)
        self.dropout_3 = nn.Dropout(0.3)
        self.conv2d_4 = nn.Conv2d(64, 32, 2, padding=1)
        self.relu_5 = nn.ReLU()
        self.linear_7 = nn.Linear(7200, 256)
        self.relu_8 = nn.ReLU()
        self.dropout_9 = nn.Dropout(0.5)
        self.linear_10 = nn.Linear(256, 10)

    def forward(self, inp):
        res = self.conv2d_0(inp)
        res = self.relu_1(inp)
        res = self.pool_2(inp)
        res = self.dropout_3(inp)
        res = self.conv2d_4(inp)
        res = self.relu_5(inp)
        res = res.reshape((7200,))
        res = self.linear_7(res)
        res = self.relu_8(res)
        res = self.dropout_9(res)
        res = self.linear_10(res)
        return res

def model_torchluent():
    return (
        FluentModule((1, 28, 28))  # features, height, width
        .verbose() # optional
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
        # we omit softmax as that's typically part of the loss
        .build()
    )



if __name__ == '__main__':
    print(model_torchluent())
    print()
    print()
    print(ModelPureTorch())
