from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py
from .basemapper import BaseMapper

np.random.seed(123)
npa = np.array


# TF implementation of RF limited Regression


class MultiMapper(BaseMapper):
  def __init__(self, graph=None, num_neurons=65, batch_size=50, init_lr=0.01,
               ls=0.05, ld=0.1, tol=1e-2, max_epochs=10, map_type='linreg', inits=None,
               log_rate=100, decay_rate=200, gpu_options=None):
    """
    Mapping function class.
    :param graph: tensorflow graph to build the mapping function with
    :param num_neurons: number of neurons (response variable) to predict
    :param batch_size: batch size
    :param init_lr: initial learning rate
    :param ls: regularization coefficient for spatial parameters
    :param ld: regularization coefficient for depth parameters
    :param tol: tolerance - stops the optimization if reaches below tol
    :param max_epochs: maximum number of epochs to train
    :param map_type: type of mapping function ('linreg', 'separable')
    :param inits: initial values for the mapping function parameters. A dictionary containing
                  any of the following keys ['s_w', 'd_w', 'bias']
    :param log_rate: rate of logging the loss values
    :param decay_rate: rate of decay for learning rate (#epochs)
    """

    super(MultiMapper, self).__init__(graph=graph, num_neurons=num_neurons, batch_size=batch_size, init_lr=init_lr,
                                      ls=ls, ld=ld, tol=tol, max_epochs=max_epochs, map_type=map_type, inits=inits,
                                      log_rate=log_rate, decay_rate=decay_rate, gpu_options=gpu_options)

  def _make_separable_map(self):
    """
    Makes the mapping function computational graph
    :return:
    """
    # use the class Mapper to map to multiple layers
    # - make sure all the feature map sizes are similar
    # - define W_s per layer features
    # - multiply by the spatial mask variable (similar to Mapper)
    # - multiply by the depthwise mixing vector (similar to Mapper but the depth is the sum of all filters in all layers)

  def _make_loss(self):
    """
    Makes the loss computational graph
    :return:
    """
    # make sure that all the variables in all mapping functions are considered in the regularization term (look at Mapper)

  def fit(self, X, Y):
    """
    Fits the parameters to the data
    :param X: Input data, first dimension is examples
    :param Y: response values (neurons), first dimension is examples
    :return:
    """
    with self._graph.as_default():
      if self._map_type == 'linreg':
        assert X.ndim == 2, 'Input matrix rank should be 2.'
      else:
        assert X.ndim == 4, 'Input matrix rank should be 4.'
      if self._is_initialized is False:
        self._init_mapper(X)

      for e in range(self._max_epochs):
        for counter, batch in enumerate(self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
          feed_dict = {self._input_ph: batch[0],
                       self.target_ph: batch[1],
                       self._lr_ph: self._lr}
          _, loss_value, reg_loss_value = self._sess.run([self.train_op, self.l2_error, self.reg_loss],
                                                         feed_dict=feed_dict)
        if (e % self._log_rate == 0) or (e == self._max_epochs - 1):
          print('Epoch: %d, Err Loss: %.2f, Reg Loss: %.2f' % (e + 1, loss_value, reg_loss_value))
        if e % self._decay_rate == 0 and e != 0:
          self._lr /= 10.
        if loss_value < self._tol:
          print('Converged.')
          break

  def predict(self, X):
    """
    Predicts the respnoses to the give input X
    :param X: Input data, first dimension is examples
    :return: predictions
    """
    with self._graph.as_default():
      if self._is_initialized is False:
        self._init_mapper(X)

      preds = []
      for batch in self._iterate_minibatches(X, batchsize=1, shuffle=False):
        feed_dict = {self._input_ph: batch}
        preds.append(np.reshape(self._sess.run([self._predictions], feed_dict=feed_dict),
                                newshape=(1, -1)))
      return np.concatenate(preds, axis=0)

  def save_weights(self, save_path):
    """
    Save weights to an hdf5 file
    :param save_path: save path
    :return:
    """
    # make sure to save weights for all the mapping functions

  def _init_mapper(self, X):
    """
    Initializes the mapping function graph
    :param X: input data
    :return:
    """
    # initialize all mapping functions
