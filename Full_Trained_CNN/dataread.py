import os
import time
import numpy as np
import tensorflow as tf
from cnn_model import MNISTcnn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

def batch_iter(data, batch_size):
		""" Generates a batch iterator for a dataset."""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int((len(data)-1)/batch_size)
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_data = data[shuffle_indices]

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = (batch_num + 1) * batch_size
			yield shuffled_data[start_index:end_index]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def main():
	num_epochs = 10
	batch_size = 100
	learning_rate = 0.001
	num_classes = 2
	label_encoder = LabelEncoder()

	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images  # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	train_data = train_data[(train_labels==1) | (train_labels==2)]
	train_labels = train_labels[(train_labels==1) | (train_labels==2)]

	train_data = train_data.reshape(-1, 28, 28, 1)
	train_labels = label_encoder.fit_transform(train_labels) # to convert labels in range (0, num_classes) for one hot encoding to happen properly
	
	train_labels = dense_to_one_hot(train_labels, num_classes)

	eval_data = mnist.test.images  # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	eval_data = eval_data[(eval_labels==1) | (eval_labels==2)]
	eval_labels = eval_labels[(eval_labels==1) | (eval_labels==2)]

	eval_data = eval_data.reshape(-1, 28, 28, 1)
	eval_labels = label_encoder.fit_transform(eval_labels)
	
	eval_labels = dense_to_one_hot(eval_labels, num_classes)

	model = MNISTcnn(learning_rate = learning_rate, num_classes = num_classes)

	saver = tf.train.Saver(tf.trainable_variables())

	with tf.Session() as sess:
		print('Starting training')
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):

			begin  = time.time()
			train_accuracies = []
			train_batches = batch_iter(list(zip(train_data, train_labels)), batch_size)
			for batch in train_batches:
				x_batch, y_batch = zip(*batch)
				feed_dict = {model.X: x_batch, model.Y_: y_batch}

				training_optimizer, cross_entropy, accuracy = sess.run([model.training_optimizer, model.cross_entropy, model.accuracy], feed_dict)
				train_accuracies.append(accuracy)

			train_acc_mean = np.mean(train_accuracies)

			test_accuracies = []
			test_batches = batch_iter(list(zip(eval_data, eval_labels)), batch_size)
			for batch in test_batches:
				x_batch, y_batch = zip(*batch)
				feed_dict = {model.X: x_batch, model.Y_: y_batch}

				training_optimizer, cross_entropy, accuracy = sess.run([model.training_optimizer, model.cross_entropy, model.accuracy], feed_dict)
				test_accuracies.append(accuracy)
			test_acc_mean = np.mean(test_accuracies)
			print("Epoch %d, time = %ds, train accuracy = %.4f, testing accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean, test_acc_mean))


		feed_dict = {model.X: eval_data, model.Y_: eval_labels}

		checkY, accuracy = sess.run([model.checkY, model.accuracy], feed_dict)
		print("Final testing accuracy ", accuracy)
		print(confusion_matrix(np.argmax(eval_labels, axis = 1), np.argmax(checkY, axis = 1)))

		ckpt_file = './../checkpoints/mnist_model.ckpt'
		saver.save(sess, ckpt_file)


if __name__ == "__main__":
    main()
