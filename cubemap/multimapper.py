from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py
from .basemapper import BaseMapper
from .mapper import Mapper

np.random.seed(123)
npa = np.array


# TF implementation of RF limited Regression


class MultiMapper(BaseMapper):
  def __init__(self, graph=None, num_readout_layers=1, num_neurons=65, batch_size=50, init_lr=0.01,
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
    self._num_readout_layers = num_readout_layers
    super(MultiMapper, self).__init__(graph=graph, num_neurons=num_neurons, batch_size=batch_size, init_lr=init_lr,
                                      ls=ls, ld=ld, tol=tol, max_epochs=max_epochs, map_type=map_type, inits=inits,
                                      log_rate=log_rate, decay_rate=decay_rate, gpu_options=gpu_options)

    self._mappers = []
    for _ in range(self._num_readout_layers):
      self._mappers.append(Mapper(graph=self._graph, num_neurons=num_neurons, batch_size=batch_size, init_lr=None,
                                  ls=ls, ld=ld, tol=tol, max_epochs=max_epochs, map_type=map_type, inits=inits,
                                  log_rate=log_rate, decay_rate=decay_rate, gpu_options=gpu_options, multimode=True))

  def _iterate_minibatches(self, inputs, targets=None, batchsize=128, shuffle=False):
    """
    Iterates over inputs with minibatches
    :param inputs: input dataset, first dimension should be examples
    :param targets: [n_examples, n_neurons] response values, first dimension should be examples
    :param batchsize: batch size
    :param shuffle: flag indicating whether to shuffle the data while making minibatches
    :return: minibatch of (X, Y)
    """
    assert self._num_readout_layers == len(inputs)
    assert sum([len(inputs[i]) == len(inputs[i+1]) for i in range(len(inputs) - 1)]) == self._num_readout_layers - 1

    input_len = inputs[0].shape[0]
    if shuffle:
      indices = np.arange(input_len)
      np.random.shuffle(indices)
    for start_idx in range(0, input_len // batchsize * batchsize, batchsize):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
      else:
        excerpt = slice(start_idx, start_idx + batchsize)
      if targets is None:
        yield [inputs[i][excerpt] for i in range(self._num_readout_layers)]
      else:
        yield [inputs[i][excerpt] for i in range(self._num_readout_layers)], targets[excerpt]

  def _make_separable_map(self):
    """
    Makes the mapping function computational graph
    :return:
    """
    for i in range(self._num_readout_layers):
      self._mappers[i]._make_separable_map(scope=f'tap{i}')
      assert self._sess is not None, 'Session is not initialized yet.'
      self._mappers[i]._sess = self._sess
      self._mappers[i]._is_initialized = True
    self._predictions = tf.reduce_mean([m._predictions for m in self._mappers], axis=0)

  def _make_loss(self):
    """
    Makes the loss computational graph
    :return:
    """
    self.reg_loss = 0
    self.l2_error = 0
    for i in range(self._num_readout_layers):
      self._mappers[i]._make_loss()
      self.reg_loss += self._mappers[i].reg_loss
      # self.l2_error += self._mappers[i].l2_error
    self.l2_error = tf.norm(self._predictions - self.target_ph, ord=2)
    self.total_loss = self.l2_error + self.reg_loss
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                       global_step=tf.train.get_or_create_global_step())

  def fit(self, X, Y):
    """
    Fits the parameters to the data
    :param X: Input data, first dimension is examples
    :param Y: response values (neurons), first dimension is examples
    :return:
    """
    with self._graph.as_default():
      if self._map_type == 'linreg':
        assert X[0].ndim == 2, 'Input matrix rank should be 2.'
      else:
        assert X[0].ndim == 4, 'Input matrix rank should be 4.'
      if self._is_initialized is False:
        self._init_mapper(X)

      for e in range(self._max_epochs):
        for counter, batch in enumerate(self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
          feed_dict = {p: batch[0][i] for i, p in enumerate(self.input_phs)}
          feed_dict.update({self.target_ph: batch[1], self._lr_ph: self._lr})
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
        feed_dict = {p: batch[i] for i, p in enumerate(self.input_phs)}
        preds.append(np.reshape(self._sess.run([self._predictions], feed_dict=feed_dict),
                                newshape=(1, -1)))
      return np.concatenate(preds, axis=0)

  def save_weights(self, save_path):
    """
    Save weights to an hdf5 file
    :param save_path: save path
    :return:
    """
    ws = {}
    for i in range(self._num_readout_layers):
      ws[self._mappers[i]._scope] = self._mappers[i].get_weights()

    print('Opening file to write to...')
    with h5py.File(save_path, 'w') as h5file:
      for k, v in ws.items():
        if 'separable' in self._map_type:
            h5file.create_dataset(f'{k}/s_w', data=v['s_w'])
            h5file.create_dataset(f'{k}/d_w', data=v['d_w'])
            h5file.create_dataset(f'{k}/bias', data=v['bias'])
        else:
          h5file.create_dataset(f'{k}/w', data=v['w'])
          h5file.create_dataset(f'{k}/bias', data=v['bias'])
    print('Finished saving.')

  def _init_mapper(self, X):
    """
    Initializes the mapping function graph
    :param X: input data
    :return:
    """
    assert hasattr(X, "__len__")
    with self._graph.as_default():
      if self._gpu_options is None:
        self._sess = tf.Session()
      else:
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=self._gpu_options))

      self.input_phs = [tf.placeholder(dtype=tf.float32, shape=[None] + list(x.shape[1:])) for x in X]
      self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._num_neurons])
      for i in range(self._num_readout_layers):
        self._mappers[i]._input_ph = self.input_phs[i]
        self._mappers[i].target_ph = self.target_ph
      # Build the model graph
      self._make_separable_map()
      self._make_loss()
      assert self.all_initalized
      self._is_initialized = True

      # initialize graph
      print('Initializing...')
      init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      self._sess.run(init_op)

  @property
  def all_initalized(self):
    return all([self._mappers[i]._is_initialized for i in range(self._num_readout_layers)])


