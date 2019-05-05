import numpy as np
import tensorflow as tf
from pre_process import *

Tx = 11
Ty = 3
n_a = 64
n_classes = 256
batch_size = 10

train_X, train_Y, val_X, val_Y, test_X, test_Y = load_dataset(1)

def return_layers():
	reshaper = tf.keras.layers.Reshape((1, n_classes))
	LSTM_cell = tf.keras.layers.LSTM(n_a, return_state=True)
	denser = tf.keras.layers.Dense(n_classes, activation='softmax')
	return reshaper, LSTM_cell, denser

def model(Tx, n_a, n_classes):
	reshaper, LSTM_cell, denser = return_layers()
	X = tf.keras.layers.Input((Tx, n_classes))
	a0 = tf.keras.layers.Input((n_a,))
	c0 = tf.keras.layers.Input((n_a,))

	outputs = []
	a = a0
	c = c0

	for t in range(Tx):
		x = X
		x = tf.keras.layers.Lambda(lambda x: x[:,t,:])(x)
		x = reshaper(x)
		a,_,c = LSTM_cell(x, initial_state=[a,c])
		if t > 7:
			out = denser(a)
			outputs.append(out)

	model = tf.keras.Model(inputs=[X,a0,c0], outputs=outputs)

	return model

def return_one_hot_batch(x, y):
	x_one_hot = np.zeros((batch_size, Tx, n_classes))
	y_one_hot = np.zeros((batch_size, Ty, n_classes))
	for i in range(batch_size):
		x_one_hot[i,:] = one_hot_encode(X[i,:], n_classes)
		y_one_hot[i,:] = one_hot_encode(Y[i,:], n_classes)
	y_one_hot = y_one_hot.transpose((1,0,2))
	return x_one_hot, y_one_hot

model = model(Tx, n_a, n_classes)
opt = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

train_losses = []
val_losses = []
for epoch in range(20):
	loss = 0
	for batch in range(10240):
		X = train_X[batch*batch_size:(batch+1)*batch_size]
		Y = train_Y[batch*batch_size:(batch+1)*batch_size]
		x_one_hot, y_one_hot = return_one_hot_batch(X, Y)
		a0 = np.zeros((batch_size, n_a))
		c0 = np.zeros((batch_size, n_a))
		history = model.fit([x_one_hot,a0,c0],list(y_one_hot),epochs=1,verbose=0)
		if batch % 1024 == 0 and batch != 0:
			print(loss/(batch+1))
		loss += np.squeeze(history.history['loss'])
	train_losses.append(loss/10240)
	print('Epoch loss: ' + str(loss/10240))
	loss = 0
	for batch in range(1024):
		X = val_X[batch*batch_size:(batch+1)*batch_size]
		Y = val_Y[batch&batch_size:(batch+1)*batch_size]
		x_one_hot, y_one_hot = return_one_hot_batch(X, Y)
		a0 = np.zeros((batch_size, n_a))
		c0 = np.zeros((batch_size, n_a))
		history = model.evaluate([x_one_hot,a0,c0],list(y_one_hot),verbose=0)
		loss += np.squeeze(history[0])
	val_losses.append(loss/10240)
	print('Val loss: ' + str(loss/10240))
	model.save('model_'+str(epoch+1)+'.h5')
	np.save('train_losses_'+str(epoch+1), train_losses)
	np.save('val_losses_'+str(epoch+1), val_losses)
	model = tf.keras.models.load_model('model_'+str(epoch+1)+'.h5')
