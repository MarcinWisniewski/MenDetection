
import timeit
import datetime
import numpy
import theano
import matplotlib.pyplot as plt
import theano.tensor as T
import pickle as cPickle
from Readers.data_provider import DataProvider
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet


def start_learning(learning_rate=0.01, momentum=0.9, use_model=True, n_epochs=20,
                    n_kerns=(15, 20, 20, 30, 10), batch_size=128):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (wfdb here)

    :type n_kerns: list of ints
    :param n_kerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: number of examples in minibatch
    """
    actual_time = datetime.datetime.now().time()
    print 'algorithm started at: ', actual_time.isoformat()

    rng = numpy.random.RandomState(23455)
    dp = DataProvider(
        input_dir_person='/home/marcin/data/men_detection/',
        test_percentage_split=1, validate_percentage_split=1, batch=500)
    valid_set_x, valid_set_y = dp.get_validate_images()
    test_set_x, test_set_y = dp.get_testing_images()

    print 'Number of validation examples: ', len(valid_set_x)
    print 'Number of test examples: ', len(test_set_x)

    cnn = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=(None, 3, 256, 256),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=2, output_nonlinearity=nonlinearities.softmax,

        update=nesterov_momentum,
        update_learning_rate=learning_rate,
        update_momentum=momentum,

        regression=False,
        max_epochs=10,
        verbose=1
    )
    test_x, test_y = dp.get_batch_training_images()
    test_x = numpy.asarray(test_x, dtype=theano.config.floatX)

    test_y = numpy.asarray(test_y)

    cnn.fit(test_x, test_y)
    train_loss = numpy.array([i["train_loss"] for i in cnn.train_history_])
    valid_loss = numpy.array([i["valid_loss"] for i in cnn.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.show()
    end_time = timeit.default_timer()


if __name__ == '__main__':
    start_learning(use_model=False)

