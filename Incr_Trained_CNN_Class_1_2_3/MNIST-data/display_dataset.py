from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()
print(len(images))
