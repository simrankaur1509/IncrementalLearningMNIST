import numpy as np
import tensorflow as tf

class MNISTcnn(object):

	def __init__(self, learning_rate, num_classes):
		# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
		self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
		# correct answers will go here
		self.Y_ = tf.placeholder(tf.float32, [None, num_classes])
		# variable learning rate

		# three convolutional layers with their channel counts, and a
		# fully connected layer (tha last layer has 10 softmax neurons)
		K = 4  # first convolutional layer output depth
		L = 8  # second convolutional layer output depth
		M = 12  # third convolutional layer
		N = 200  # fully connected layer

		W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1), name="W1")  # 5x5 patch, 1 input channel, K output channels
		B1 = tf.Variable(tf.ones([K])/num_classes, name="B1")
		W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name="W2")
		B2 = tf.Variable(tf.ones([L])/num_classes, name="B2")
		W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name="W3")
		B3 = tf.Variable(tf.ones([M])/num_classes, name="B3")
		W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1), name="W4")
		B4 = tf.Variable(tf.ones([N ])/num_classes, name="B4")
		W5 = tf.Variable(tf.truncated_normal([N, num_classes], stddev=0.1), name="W5")
		B5 = tf.Variable(tf.ones([num_classes])/num_classes, name="B5")

		# The model
		stride = 1  # output is 28x28
		Y1 = tf.nn.relu(tf.nn.conv2d(self.X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
		stride = 2  # output is 14x14
		Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
		stride = 2  # output is 7x7
		Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

		# reshape the output from the third convolution for the fully connected layer
		YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

		Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
		Ylogits = tf.matmul(Y4, W5) + B5
		Y = tf.nn.softmax(Ylogits)

		self.checkY = Y
		
		# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
		# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
		# problems with log(0) which is NaN
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)
		self.cross_entropy = tf.reduce_mean(cross_entropy)*100

		# accuracy of the trained model, between 0 (worst) and 1 (best)
		correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.Y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# training step, the learning rate is a placeholder
		self.training_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

