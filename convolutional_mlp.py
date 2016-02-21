
import os
import sys
import timeit
import time
import datetime
import numpy
import theano
import theano.tensor as T
import lasagne
import pickle as cPickle
from Readers.data_provider import DataProvider
from CNN.conv_network import CNN


def start_learning(learning_rate=0.001, momentum=0.9, use_model=True, n_epochs=20,
                    n_kerns=(96, 256, 128, 128, 64), batch_size=128):
    """

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type momentum: float
    :param momentum: momentum for CNN

    :type: use_model: bool
    :param use_model: True if You want to read trained model from to file

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type n_kerns: list of ints
    :param n_kerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: number of examples in minibatch
    """

    actual_time = datetime.datetime.now().time()
    print 'algorithm started at: ', actual_time.isoformat()

    rng = numpy.random.RandomState(234555)
    dp = DataProvider(
        input_dir='/home/marcinw/data/men_detection',
        test_percentage_split=0.005, validate_percentage_split=0.005, batch=batch_size)
    print 'number of images', dp.get_number_of_images()

    # start-snippet-1
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.vector('y', dtype='int64')  # the labels are presented as 1D vector of
                                     # [int] labels
   
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    cnn = CNN(rng, x, n_kerns)
    prediction = lasagne.layers.get_output(cnn.network, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(cnn.network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    test_prediction = lasagne.layers.get_output(cnn.network)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    train_model = theano.function([x, y], loss, updates=updates)
    validate_model = theano.function([x, y], [test_loss, test_acc])

    ###############
    # TRAIN MODEL #
    ###############
    if os.path.isfile('model.bin') and use_model:
        print 'using erlier model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()
    print '... training'
    n_train_batches = dp.get_number_of_training_batches()
    n_valid_batches = dp.get_number_of_validate_batches()
    n_test_batches = dp.get_number_of_testing_batches()

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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
    best_cnn = CNN(rng, x, n_kerns)

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        cost_ij = 0
        for minibatch_index in xrange(n_train_batches):
            batch_train_set_x, batch_train_set_y = dp.get_batch_training_images()
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
                print 'cost = ', cost_ij

            if batch_train_set_x is not None and batch_train_set_y is not None:
                cost_ij += train_model(batch_train_set_x, batch_train_set_y)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = []
                valaidation_acc = []
                for valid_batch in xrange(n_valid_batches):
                    batch_valid_set_x, batch_valid_set_y = dp.get_batch_validate_images()
                    if batch_valid_set_x is not None and batch_valid_set_y is not None:
                        err, acc = validate_model(batch_valid_set_x, batch_valid_set_y)
                        validation_losses.append(err)
                        valaidation_acc.append(acc)
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

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
                    test_losses = []
                    test_acc = []
                    for test_batch in xrange(n_test_batches):
                        batch_test_set_x, batch_test_set_y = dp.get_batch_testing_images()
                        if batch_test_set_x is not None and batch_test_set_y is not None:
                            err, acc = validate_model(batch_test_set_x, batch_test_set_y)
                            test_losses.append(err)
                            test_acc.append(acc)
                    test_score =  numpy.mean(test_losses)
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
    start_learning(use_model=True)

