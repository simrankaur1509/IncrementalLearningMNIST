

Needs internet connection to download the MNIST files

1. Full_Trained_CNN: 
		(a) cnn_model.py - The file describing the CNN model
		(b) dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2. Saves checkpoints that will be used for incremental training

2. Vanilla_Trained_CNN_class_1_2_3: Regular Vanilla CNN initilialized with pre-trained_weights for class 1,2 and tested on class 1,2,3
		(a) vanilla_cnn_model.py - The file describing the CNN model
		(b) vanilla_dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2 and tested on class 1,2,3

3. Vanilla_Trained_CNN_class_1_2_3_4: Regular Vanilla CNN initilialized with pre-trained_weights for class 1,2 and tested on class 1,2,3,4
		(a) cnn_model.py - The file describing the CNN model
		(b) dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2 and tested on class 1,2,3,4

4. Incr_Trained_CNN_Class_1_2_3: The proposed CNN with revised cost function initilialized with pre-trained_weights for class 1,2 and tested on class 1,2,3
		(a) vanilla_cnn_model.py - The file describing the CNN model
		(b) vanilla_dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2 and tested on class 1,2,3

5. Incr_Trained_CNN_Class_1_2_3_4: The proposed CNN with revised cost function initilialized with pre-trained_weights for class 1,2 and tested on class 1,2,3,4
		(a) cnn_model.py - The file describing the CNN model
		(b) dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2 and tested on class 1,2,3,4

6. Exp_Incr_Trained_CNN_Class_1_2_3: Testing the CNN without incorporating loss on old class images (only using new class images and weight norm)
		(a) cnn_model.py - The file describing the CNN model
		(b) dataread.py - The file used for reading MNIST data and using cnn_model.py to train a network on class 1,2 and tested on class 3