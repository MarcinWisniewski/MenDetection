__author__ = 'Marcin'

from conv_layer import LeNetConvPoolLayer
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
import theano
import theano.tensor as T


class CNN(object):
    def __init__(self, rng, input, n_kerns, batch_size):

        self.layer0_input = input

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (256-9+1 , 256-9+1) = (248, 248)
        # maxpooling reduces this further to (248/2, 248/2) = (124, 124)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 124, 124)

        self.layer0_pooling_factor = (2, 2)
        self.layer0_filter_shape = (n_kerns[0], 3, 9, 9)
        print 'layer 0 input: ', (batch_size, 3, 256, 256)
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, 3, 256, 256),
            filter_shape=self.layer0_filter_shape,
            poolsize=self.layer0_pooling_factor
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (124-5+1, 124-5+1) = (120, 120)
        # maxpooling reduces this further to (120/2, 120/2) = (60, 60)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 60, 60)
        self.layer1_pooling_factor = (2, 2)
        self.layer1_filter_shape = (n_kerns[1], n_kerns[0], 5, 5)
        self.layer1_input_shape = (batch_size, n_kerns[0], 124, 124)
        print 'layer 1 input: ', self.layer1_input_shape
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=self.layer1_input_shape,
            filter_shape=self.layer1_filter_shape,
            poolsize=self.layer1_pooling_factor
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (60-5+1, 60-5+1) = (56, 56)
        # maxpooling reduces this further to (56/2, 56/2) = (28, 28)
        # 4D output tensor is thus of shape (batch_size, nkerns[2], 28, 28)
        self.layer2_pooling_factor = (2, 2)
        self.layer2_filter_shape = (n_kerns[2], n_kerns[1], 5, 5)
        self.layer2_input_shape = (batch_size, n_kerns[1], 60, 60)
        print 'layer 2 input: ', self.layer2_input_shape
        self.layer2 = LeNetConvPoolLayer(
            rng,
            input=self.layer1.output,
            image_shape=self.layer2_input_shape,
            filter_shape=self.layer2_filter_shape,
            poolsize=self.layer2_pooling_factor
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (28-5+1, 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[3], 12, 12)
        self.layer3_pooling_factor = (2, 2)
        self.layer3_filter_shape = (n_kerns[3], n_kerns[2], 5, 5)
        self.layer3_input_shape = (batch_size, n_kerns[2], 28, 28)
        print 'layer 3 input: ', self.layer3_input_shape
        self.layer3 = LeNetConvPoolLayer(
            rng,
            input=self.layer2.output,
            image_shape=self.layer3_input_shape,
            filter_shape=self.layer3_filter_shape,
            poolsize=self.layer3_pooling_factor
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[2] * 28 * 28),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer4_input = self.layer3.output.flatten(2)

        print 'layer 4 input: ', n_kerns[3] * 12 * 12
        print 'layer 4 output: ', (n_kerns[3] * 12 * 12)/2
        # construct a fully-connected sigmoidal layer
        self.layer4 = HiddenLayer(
            rng,
            input=self.layer4_input,
            n_in=n_kerns[3] * 12 * 12,
            n_out=(n_kerns[3] * 12 * 12)/2,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer5_input_length = (n_kerns[3] * 12 * 12)/2
        print 'layer 5 input: ', self.layer5_input_length
        self.layer5 = LogisticRegression(input=self.layer4.output,
                                         n_in=self.layer5_input_length,
                                         n_out=2)

        self.errors = self.layer5.errors

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer5.params + self.layer4.params + self.layer3.params + \
                      self.layer2.params + self.layer1.params + self.layer0.params

    def __getstate__(self):
        weights = [parameter.get_value() for parameter in self.params]
        return weights

    def __setstate__(self, weights):
        weight = iter(weights)
        for parameter in self.params:
            parameter.set_value(weight.next())

    def gradient_updates_momentum(self, cost, learning_rate, momentum):
        '''
        Compute updates for gradient descent with momentum

        :parameters:
            - cost : theano.tensor.var.TensorVariable
                Theano cost function to minimize
            - params : list of theano.tensor.var.TensorVariable
                Parameters to compute gradient against
            - learning_rate : float
                Gradient descent learning rate
            - momentum : float
                Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

        :returns:
            updates : list
                List of updates, one for each parameter
        '''
        # Make sure momentum is a sane value
        assert momentum < 1 and momentum >= 0
        # List of update steps for each parameter
        updates = []
        # Just gradient descent on cost
        for param in self.params:
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            updates.append((param, param - learning_rate*param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates
