'''
1,2 then 3
'''
import numpy as np
import tensorflow as tf

class MNISTcnn(object):

	def __init__(self, learning_rate, num_classes, newW1, newW2, newW3, newW4, newW5, newB1, newB2, newB3, newB4, newB5):
		# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
		self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
		# correct answers will go here
		self.Y_ = tf.placeholder(tf.float32, [None, num_classes])

		self.old_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
		self.old_Y_ = tf.placeholder(tf.float32, [None, num_classes])

		# three convolutional layers with their channel counts, and a
		# fully connected layer (tha last layer has 10 softmax neurons)
		K = 4  # first convolutional layer output depth
		L = 8  # second convolutional layer output depth
		M = 12  # third convolutional layer
		N = 200  # fully connected layer
		lbda = 0.3

		self.old_W1 = tf.placeholder(tf.float32, [5, 5, 1, K])  # 5x5 patch, 1 input channel, K output channels
		self.old_B1 = tf.placeholder(tf.float32, [K])
		self.old_W2 = tf.placeholder(tf.float32, [5, 5, K, L])
		self.old_B2 = tf.placeholder(tf.float32, [L])
		self.old_W3 = tf.placeholder(tf.float32, [4, 4, L, M])
		self.old_B3 = tf.placeholder(tf.float32, [M])

		self.old_W4 = tf.placeholder(tf.float32, [7 * 7 * M, N])
		self.old_B4 = tf.placeholder(tf.float32, [N])
		self.old_W5 = tf.placeholder(tf.float32, [N, num_classes-1])
		self.old_B5 = tf.placeholder(tf.float32, [num_classes-1])

		self.W1 = tf.Variable(newW1)
		self.B1 = tf.Variable(newB1)
		self.W2 = tf.Variable(newW2)
		self.B2 = tf.Variable(newB2)
		self.W3 = tf.Variable(newW3)
		self.B3 = tf.Variable(newB3)

		self.W4 = tf.Variable(newW4)
		self.B4 = tf.Variable(newB4)
		self.W5 = tf.Variable(newW5)
		self.B5 = tf.Variable(newB5)

		# Foward pass for new images
		stride = 1  # output is 28x28
		Y1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, stride, stride, 1], padding='SAME') + self.B1)
		stride = 2  # output is 14x14
		Y2 = tf.nn.relu(tf.nn.conv2d(Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME') + self.B2)
		stride = 2  # output is 7x7
		Y3 = tf.nn.relu(tf.nn.conv2d(Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME') + self.B3)

		# reshape the output from the third convolution for the fully connected layer
		YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

		Y4 = tf.nn.relu(tf.matmul(YY, self.W4) + self.B4)
		Ylogits = tf.matmul(Y4, self.W5) + self.B5
		Y = tf.nn.softmax(Ylogits)

		# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
		# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
		# problems with log(0) which is NaN
		self.cost_new = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)
		self.cost_new = tf.reduce_mean(self.cost_new)#*lbda


		# Foward pass for old images
		stride = 1  # output is 28x28
		old_Y1 = tf.nn.relu(tf.nn.conv2d(self.old_image, self.W1, strides=[1, stride, stride, 1], padding='SAME') + self.B1)
		stride = 2  # output is 14x14
		old_Y2 = tf.nn.relu(tf.nn.conv2d(old_Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME') + self.B2)
		stride = 2  # output is 7x7
		old_Y3 = tf.nn.relu(tf.nn.conv2d(old_Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME') + self.B3)

		# reshape the output from the third convolution for the fully connected layer
		old_YY = tf.reshape(old_Y3, shape=[-1, 7 * 7 * M])

		old_Y4 = tf.nn.relu(tf.matmul(old_YY, self.W4) + self.B4)
		old_Ylogits = tf.matmul(old_Y4, self.W5) + self.B5
		old_Y = tf.nn.softmax(old_Ylogits)

		# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
		# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
		# problems with log(0) which is NaN
		self.cost_old = tf.nn.softmax_cross_entropy_with_logits(logits=old_Ylogits, labels=self.old_Y_)
		self.cost_old = tf.reduce_mean(self.cost_old)

		weight_diff = tf.add_n([tf.norm(tf.subtract(self.W1, self.old_W1), ord=1) , tf.norm(tf.subtract(self.W2, self.old_W2), ord=1) ,
		tf.norm(tf.subtract(self.W3, self.old_W3), ord=1) , tf.norm(tf.subtract(self.W4, self.old_W4), ord=1) ,
		tf.norm(tf.subtract(self.W5[:, :num_classes-1], self.old_W5), ord=1)])

		bias_diff = tf.add_n([tf.norm(tf.subtract(self.B1, self.old_B1), ord=1) , tf.norm(tf.subtract(self.B2, self.old_B2), ord=1) ,
		tf.norm(tf.subtract(self.B3, self.old_B3), ord=1) , tf.norm(tf.subtract(self.B4, self.old_B4), ord=1) ,
		tf.norm(tf.subtract(self.B5[:num_classes-1], self.old_B5), ord=1)])

		self.overall_weight_diff = tf.add(weight_diff , bias_diff)
		self.term2 = self.overall_weight_diff#*(1-lbda)
		# self.term2 = tf.multiply(self.cost_old, self.overall_weight_diff)#*(1-lbda)

		self.cost_total = tf.add(self.cost_new, self.term2)
		
		# # accuracy of the trained model, between 0 (worst) and 1 (best)
		self.correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.Y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.checkY = Y

		self.training_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost_total)
		
		# removed lamba, and then try with cost_old present/ unpresent
