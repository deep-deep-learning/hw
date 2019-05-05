import numpy as np
import tensorflow as tf

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		batch = pickle.load(fo, encoding='latin1')
	return batch

def unpickle_batch(dir):
	batch = unpickle(dir)
	data = batch['data']
	labels = batch['labels']
	return data, labels

def load_batch(dir):
	data, labels = unpickle_batch(dir)
	data = data.reshape((len(data), 3, 32, 32)).transpose(0, 2, 3, 1)
	data = np.pad(data, ((0,0), (1,1), (1,1), (0,0)), 'constant', constant_values=((0,0), (0,0), (0,0), (0,0)))
	return data, labels

def make_sequence(data):
	X = np.zeros((32*32,10000,11))
	Y = np.zeros((32*32,10000,3))
	for i in range(32):
		for j in range(32):
			sequence = np.dstack((data[:,i,j,0], data[:,i,j,1], data[:,i,j,2], data[:,i,j+1,0], data[:,i,j+1,1], data[:,i,j+1,2], data[:,i+1,j,0], data[:,i+1,j,1], data[:,i+1,j,2], data[:,i+1,j+1,0], data[:,i+1,j+1,1]))
			X[i+j] = sequence
			sequence = np.dstack((data[:,i+1,j+1,0], data[:,i+1,j+1,1], data[:,i+1,j+1,2]))
			Y[i+j] = sequence
	X = X.reshape(-1, X.shape[2]).astype('int32')
	Y = Y.reshape(-1, Y.shape[2]).astype('int32')
	return X, Y

def one_hot_encode(x, num_classes):
	one_hot = np.zeros((x.shape[0], num_classes))
	one_hot[np.arange(x.shape[0]), x] = 1
	return one_hot

def load_dataset(num_batches):
	train_X = np.zeros((10240000*num_batches, 11), dtype='int32')
	train_Y = np.zeros((10240000*num_batches, 3), dtype='int32')
	for i in range(num_batches):
		data, labels = load_batch('cifar-10-batches-py/data_batch_'+str(i+1))
		X, Y = make_sequence(data)	
		train_X[10240000*i:10240000*(i+1)] = X
		train_Y[10240000*i:10240000*(i+1)] = Y

	data, labels = load_batch('cifar-10-batches-py/test_batch')
	test_X, test_Y = make_sequence(data)
	val_X = test_X[102400:]
	val_Y = test_Y[102400:]
	test_X = test_X[102400:204800]
	test_Y = test_Y[102400:204800]
	return train_X, train_Y, val_X, val_Y, test_X, test_Y

def save_dataset():
	train_X, train_Y, val_X, val_Y, test_X, test_Y = load_dataset()
	one_hot = np.zeros((46080000,11,256))
	for i in range(11):
		one_hot[np.arange(46080000),i,train_X[:,i]] = 1
	train_X = one_hot
	one_hot = np.zeros((46080000,3,256))
	for i in range(3):
		one_hot[np.arange(46080000),i,train_Y[:,i]] = 1
	train_Y = one_hot
	one_hot = np.zeros((5120000,11,256))
	for i in range(11):
		one_hot[np.arange(5120000),i,val_X[:,i]] = 1
	val_X = one_hot
	one_hot = np.zeros((5120000,3,256))
	for i in range(3):
		one_hot[np.arange(5120000),i,val_Y[:,i]] = 1
	val_Y = one_hot
	one_hot = np.zeros((10240000,11,256))
	for i in range(11):
		one_hot[np.arange(10240000),i,test_X[:,i]] = 1
	test_X = one_hot
	one_hot = np.zeros((10240000,11,256))
	for i in range(3):
		one_hot[np.arange(10240000),i,test_Y[:,i]] = 1
	test_Y = one_hot
	print(train_X.shape)
	print(train_Y.shape)
	print(val_X.shape)
	print(val_Y.shape)
	print(test_X.shape)
	print(test_Y.shape)

