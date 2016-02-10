__author__ = 'Marcin'
import lasagne


class CNN(object):
    def __init__(self, rng, input, n_kerns):

        self.layer0_input = input

        self.network = lasagne.layers.InputLayer(shape=(None, 3, 233, 233),
                                                 input_var=self.layer0_input)

        self.network = lasagne.layers.Conv2DLayer(incoming=self.network, num_filters=n_kerns[0],
                                                  filter_size=(11, 11), stride=(4, 4),
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.Conv2DLayer(incoming=self.network, num_filters=n_kerns[1],
                                                  filter_size=(5, 5),
                                                  nonlinearity=lasagne.nonlinearities.rectify)
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))

        self.network = lasagne.layers.Conv2DLayer(incoming=self.network, num_filters=n_kerns[2],
                                                  filter_size=(3, 3),
                                                  nonlinearity=lasagne.nonlinearities.rectify)
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))

        self.network = lasagne.layers.Conv2DLayer(incoming=self.network, num_filters=n_kerns[3],
                                                  filter_size=(3, 3),
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.Conv2DLayer(incoming=self.network, num_filters=n_kerns[4],
                                                  filter_size=(3, 3),
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=.5),
                                                 num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=.5),
                                                 num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.DenseLayer(self.network, num_units=2,
                                                 nonlinearity=lasagne.nonlinearities.softmax)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        self.prediction = lasagne.layers.get_output(self.network)

    def __getstate__(self):
        return lasagne.layers.get_all_param_values(self.network)

    def __setstate__(self, weights):
        lasagne.layers.set_all_param_values(self.network, weights)
