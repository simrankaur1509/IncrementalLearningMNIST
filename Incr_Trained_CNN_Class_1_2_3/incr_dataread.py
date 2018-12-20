import os
import time
import skimage
import numpy as np
from math import pi
import tensorflow as tf
import matplotlib.pyplot as plt
from incr_cnn_model import MNISTcnn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

np.random.seed(9001)

def batch_iter(data, batch_size):
		""" Generates a batch iterator for a dataset."""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int((len(data)-1)/batch_size)  
		shuffle_indices = np.random.permutation(np.arange(data_size)) #Returns evenly spaced numbers np.arange(3)=[0 1 2]
		shuffled_data = data[shuffle_indices]

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = (batch_num + 1) * batch_size
			yield shuffled_data[start_index:end_index] #yield acts like 'return' but it can return multiple values 

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, 28, 28, 1))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    print(X_rotate.shape)
    return X_rotate

def add_salt_pepper_noise(X_imgs): 
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    copy = []
    for X_img in X_imgs_copy:
    	X_img = skimage.util.random_noise(X_img, mode='gaussian', seed=None, clip=True)
    	copy.append(X_img)
    print(np.asarray(copy).shape, X_imgs.shape)
    return np.asarray(copy)


def main():
	num_epochs = 10
	batch_size = 100
	learning_rate = 1e-4
	num_classes = 3
	label_encoder = LabelEncoder()

	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images  # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	original_labels = train_labels[(train_labels==1) | (train_labels==2) | (train_labels==3)]
	label_encoder = label_encoder.fit(original_labels)

	class_1_data = train_data[(train_labels==1)][0:3] # improve ths later. First 3 images of class 1
	class_1_data = class_1_data.reshape(-1, 28, 28, 1)
	# class_1_data = np.concatenate((class_1_data, rotate_images(class_1_data, 0, 5, 20)))
	# class_1_data = np.concatenate((class_1_data, add_salt_pepper_noise(class_1_data)))

	class_2_data = train_data[(train_labels==2)][0:3] #improve this later. First 3 images of class 2
	class_2_data = class_2_data.reshape(-1, 28, 28, 1)
	# class_2_data = np.concatenate((class_2_data, rotate_images(class_2_data, 0, 5, 20)))
	# class_2_data = np.concatenate((class_2_data, add_salt_pepper_noise(class_2_data)))

	class_1_label = train_labels[(train_labels==1)][0:3] # improve ths later. First 3 images of class 1
	# for i in range(0, 20):
	# 	class_1_label = np.concatenate((class_1_label, train_labels[(train_labels==1)][0:3]))

	class_2_label = train_labels[(train_labels==2)][0:3] #improve this later. First 3 images of class 2
	# for i in range(0, 20):
	# 	class_2_label = np.concatenate((class_2_label, train_labels[(train_labels==2)][0:3]))


	class_data = np.concatenate((class_1_data, class_2_data))
	class_data = class_data.reshape(-1, 28, 28, 1)
	class_label = np.concatenate((class_1_label, class_2_label))
	class_label = label_encoder.transform(class_label) # to convert labels in range (0, num_classes) for one hot encoding to happen properly
	class_label = dense_to_one_hot(class_label, num_classes)

	new_train_data = train_data[(train_labels==3)]
	new_train_labels = train_labels[(train_labels==3)]

	new_train_data = new_train_data.reshape(-1, 28, 28, 1)
	new_train_labels = label_encoder.transform(new_train_labels) # to convert labels in range (0, num_classes) for one hot encoding to happen properly
	
	new_train_labels = dense_to_one_hot(new_train_labels, num_classes)

	eval_data = mnist.test.images  # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	eval_data = eval_data[(eval_labels==1) | (eval_labels==2) | (eval_labels==3)]
	eval_labels = eval_labels[(eval_labels==1) | (eval_labels==2) | (eval_labels==3)]

	eval_data = eval_data.reshape(-1, 28, 28, 1)
	eval_labels = label_encoder.transform(eval_labels)
	
	eval_labels = dense_to_one_hot(eval_labels, num_classes)

	dummy_W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
	dummy_B1 = tf.Variable(tf.ones([4])/num_classes-1)
	dummy_W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
	dummy_B2 = tf.Variable(tf.ones([8])/num_classes-1)
	dummy_W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
	dummy_B3 = tf.Variable(tf.ones([12])/num_classes-1)

	dummy_W4 = tf.Variable(tf.truncated_normal([7 * 7 * 12, 200], stddev=0.1))
	dummy_B4 = tf.Variable(tf.ones([200])/num_classes-1)
	dummy_W5 = tf.Variable(tf.truncated_normal([200, num_classes-1], stddev=0.1))
	dummy_B5 = tf.Variable(tf.ones([num_classes-1])/num_classes-1)

	saver = tf.train.Saver({"W1": dummy_W1, "W2": dummy_W2, "W3": 
		dummy_W3, "W4": dummy_W4, "W5": dummy_W5, "B1": dummy_B1,
		 "B2": dummy_B2, "B3": dummy_B3, "B4": dummy_B4, "B5": dummy_B5}) # CHECK -Maps W1 to dummy W1 and so on

	with tf.Session() as sess:
		print('Starting training')
		ckpt_file = './../checkpoints/mnist_model.ckpt'
		print('Restoring parameters from', ckpt_file)
		saver.restore(sess, ckpt_file)  #Loads checkpoints
        
		newW1=dummy_W1.eval()   #eval() converts them into numeric arrays for calculation 
		newW2=dummy_W2.eval()
		newW3=dummy_W3.eval()
		newW4=dummy_W4.eval() 
		newB1=dummy_B1.eval()
		newB2=dummy_B2.eval()
		newB3=dummy_B3.eval()
		newB4=dummy_B4.eval()
                                  #W5 and B5 have things for class 3 also, needs random values
		newW5 = np.random.normal(scale=0.1, size=[200, num_classes]) # HARD CODED
		newW5[:, :num_classes-1] = dummy_W5.eval()  #copies class 1 and class 2 ke liye from checkpoints
		newB5 = np.random.normal(scale=0.1, size=[num_classes])
		newB5[:num_classes-1] = dummy_B5.eval()
		newW5 = newW5.astype(np.float32)
		newB5 = newB5.astype(np.float32)

		model = MNISTcnn(learning_rate = learning_rate, num_classes = num_classes, newW1=dummy_W1.eval(), newW2=dummy_W2.eval(),\
			newW3=dummy_W3.eval(), newW4=dummy_W4.eval(), newW5=newW5, newB1=dummy_B1.eval(), newB2=dummy_B2.eval(), \
			newB3=dummy_B3.eval(), newB4=dummy_B4.eval(),\
			newB5=newB5)   # newW4 would send it by reference. Therefore, we used newW4=dummy_W4.eval()
		sess.run(tf.global_variables_initializer())

		#W5=sess.run(model.W5)

		flag=0
		# plot_cost = []
		# cnt = 0

		for epoch in range(num_epochs):

			begin  = time.time()

			data_size = len(class_data)
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = class_data[shuffle_indices]
			shuffled_labels = class_label[shuffle_indices]

			new_train_batches = batch_iter(list(zip(new_train_data, new_train_labels)), batch_size)
			# temp_cost = []

			for batch in new_train_batches:

				x_batch, y_batch = zip(*batch)

				feed_dict = {model.X: x_batch, model.Y_: y_batch, model.old_image: shuffled_data, model.old_Y_: shuffled_labels
				,\
				model.old_W1: newW1,
				model.old_W2: newW2,
				model.old_W3: newW3,
				model.old_W4: newW4,
				model.old_W5: newW5[:, :num_classes-1],
				model.old_B1: newB1,
				model.old_B2: newB2,
				model.old_B3: newB3,
				model.old_B4: newB4,
				model.old_B5: newB5[:num_classes-1]
				}
				
				_, cost_total, cost_new, cost_old, term2 = sess.run([model.training_optimizer, model.cost_total,\
					model.cost_new, model.cost_old,model.term2], feed_dict)
				print("Loss [total, new, term2] ",cost_total, cost_new, term2)
				if(cost_total<0.1):
					flag=1
					break
				# temp_cost.append(cost_total)

			feed_dict = {model.X: eval_data, model.Y_: eval_labels}

			checkY, accuracy = sess.run([model.checkY, model.accuracy], feed_dict)

			print("Epoch ",epoch," Accuracy  ", accuracy)
			print(confusion_matrix(np.argmax(eval_labels, axis = 1), np.argmax(checkY, axis = 1)))

			if(flag==1):
				break
			# plot_cost.append(np.mean(temp_cost))
			# cnt+=1

		feed_dict = {model.X: eval_data, model.Y_: eval_labels}

		correct_prediction, checkY, accuracy = sess.run([model.correct_prediction, model.checkY, model.accuracy], feed_dict)
		
		print("Final Accuracy : ", accuracy)
		print(confusion_matrix(np.argmax(eval_labels, axis = 1), np.argmax(checkY, axis = 1)))

		# fig = plt.figure()
		# ax = plt.axes()

		# x = np.arange(cnt)
		# ax.plot(x, plot_cost)
		# plt.show()

if __name__ == "__main__":
	main()
