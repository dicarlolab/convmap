from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py


class BaseMapper(object):
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
    self._ld = ld  # reg factor for depth conv
    self._ls = ls  # reg factor for spatial conv
    self._tol = tol
    self._batch_size = batch_size
    self._num_neurons = num_neurons
    self._lr = init_lr
    self._max_epochs = max_epochs
    self._map_type = map_type
    self._inits = inits
    self._is_initialized = False
    self._log_rate = log_rate
    self._decay_rate = decay_rate
    self._gpu_options = gpu_options
    assert map_type in ['linreg', 'separable', 'separable_legacy']

    if graph is None:
      self._graph = tf.Graph()
    else:
      self._graph = graph

    with self._graph.as_default():
      self._lr_ph = tf.placeholder(dtype=tf.float32)
      self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

  def _iterate_minibatches(self, inputs, targets=None, batchsize=128, shuffle=False):
    """
    Iterates over inputs with minibatches
    :param inputs: input dataset, first dimension should be examples
    :param targets: [n_examples, n_neurons] response values, first dimension should be examples
    :param batchsize: batch size
    :param shuffle: flag indicating whether to shuffle the data while making minibatches
    :return: minibatch of (X, Y)
    """
    input_len = inputs.shape[0]
    if shuffle:
      indices = np.arange(input_len)
      np.random.shuffle(indices)
    for start_idx in range(0, input_len // batchsize * batchsize, batchsize):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
      else:
        excerpt = slice(start_idx, start_idx + batchsize)
      if targets is None:
        yield inputs[excerpt]
      else:
        yield inputs[excerpt], targets[excerpt]

  def _make_separable_map(self):
    raise NotImplementedError

  @staticmethod
  def _l2_loss(weights):
    return tf.reduce_sum(weights ** 2) / tf.to_float(weights.shape[0])

  @staticmethod
  def _l1_loss(weights):
    return tf.reduce_sum(tf.abs(weights)) / tf.to_float(weights.shape[0])

  def _make_loss(self):
    raise NotImplementedError

  def fit(self, X, Y):
    raise NotImplementedError

  def predict(self, X):
    raise NotImplementedError

  def save_weights(self, save_path):
    raise NotImplementedError

  def close(self):
    """
    Closes occupied resources
    :return:
    """
    tf.reset_default_graph()
    self._sess.close()
