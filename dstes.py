import numpy as np
from tensorflow import keras
inputs = np.random.random((32, 10, 8))
lstm = keras.layers.LSTM(4)
output = lstm(inputs)
