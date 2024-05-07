import pandas as pd
import numpy as np

# Make numpy values easier to read
# from keras.src.legacy import layers
from tensorflow.python.layers import layers

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

abalone_train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
                            names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                                   "Viscera weight", "Shell weight", "Age"])

abalone_train.head()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
print(abalone_features)

# A regression model predict the age. Since there is only a single input tensor, a tf.keras.Sequential model is
# sufficient here.

abalone_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)

normalize = tf.keras.layers.Normalization()
normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
    normalize,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

norm_abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
