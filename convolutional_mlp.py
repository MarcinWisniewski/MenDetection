"""This algorithm implements CNN for ECG analysis

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit
import datetime
import numpy
import theano
import theano.tensor as T
import pickle as cPickle
from Readers.data_provider import DataProvider
from CNN.conv_network import CNN


def start_learning(learning_rate=0.01, momentum=0.9, use_model=True, n_epochs=20,
                    n_kerns=(10, 15, 20, 20), batch_size=128):
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
        input_dir_person='/home/marcin/data/men_detection/men',
        input_dir_background='/home/marcin/data/men_detection/mountains',
        test_percentage_split=5, validate_percentage_split=5, batch=batch_size)
    valid_set_x, valid_set_y = dp.get_validate_images()
    test_set_x, test_set_y = dp.get_testing_images()

    n_valid_batches = len(valid_set_x)/batch_size - 1
    n_test_batches = len(test_set_x)/batch_size - 1

    valid_set_x = theano.shared(numpy.asarray(valid_set_x, dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(valid_set_y, dtype='int8'), borrow=True)

    test_set_x = theano.shared(numpy.asarray(test_set_x, dtype=theano.config.floatX), borrow=True)
    test_set_y = theano.shared(numpy.asarray(test_set_y, dtype='int8'), borrow=True)



    # start-snippet-1
    index = T.lscalar('index')
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.vector('y', dtype='int8')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    cnn = CNN(rng, x, n_kerns, batch_size)
    # the cost we minimize during training is the NLL of the model
    cost = cnn.layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        cnn.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        cnn.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    train_model = theano.function([x, y], cost, updates=cnn.gradient_updates_momentum(cost,  learning_rate, momentum))

    ###############
    # TRAIN MODEL #
    ###############
    if os.path.isfile('model.bin') and use_model:
        print 'using erlier model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()
    print '... training'
    n_train_batches = dp.get_number_of_batches()
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    print 'validation frequency:', validation_frequency

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    best_cnn = CNN(rng, x, n_kerns, batch_size)

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        dp.clear_batch_index()
        for minibatch_index in xrange(n_train_batches):
            batch_train_set_x, batch_train_set_y = dp.get_batch_training_images()
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 10 == 0:
                print 'training @ iter = ', iter
            if batch_train_set_x is not None and batch_train_set_y is not None:
                cost_ij = train_model(batch_train_set_x, batch_train_set_y)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    best_weights = cnn.__getstate__()
                    best_cnn.__setstate__(best_weights)
                    f = open('model.bin', 'wb')
                    cPickle.dump(best_cnn.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    start_learning()

