import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
train_losses = np.load('train_losses.npy')
val_losses = np.load('val_losses.npy')
