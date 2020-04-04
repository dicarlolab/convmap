from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py
from .basemapper import BaseMapper

np.random.seed(123)
npa = np.array


# TF implementation of RF limited Regression


class Mapper(BaseMapper):
  def __init__(self, graph=None, num_neurons=65, batch_size=50, init_lr=0.01,
               ls=0.05, ld=0.1, tol=1e-2, max_epochs=10, map_type='linreg', inits=None,
               log_rate=100, decay_rate=200, gpu_options=None, multimode=False, use_l1loss=False):
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

    super(Mapper, self).__init__(graph=graph, num_neurons=num_neurons, batch_size=batch_size, init_lr=init_lr,
                                 ls=ls, ld=ld, tol=tol, max_epochs=max_epochs, map_type=map_type, inits=inits,
                                 log_rate=log_rate, decay_rate=decay_rate, gpu_options=gpu_options, multimode=multimode)
    self.use_l1loss = use_l1loss

  def _make_separable_map(self, scope='mapping'):
    """
    Makes the mapping function computational graph
    :return:
    """
    self._scope = scope
    with self._graph.as_default():
      with tf.variable_scope(scope):
        if self._map_type == 'separable':
          input_shape = self._input_ph.shape
          if self._inits is None:
            s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1, self._num_neurons),
                              dtype=tf.float32)
            d_w = tf.Variable(initial_value=np.random.randn(1, input_shape[-1], self._num_neurons), dtype=tf.float32)
            bias = tf.Variable(initial_value=np.zeros((1, self._num_neurons)), dtype=tf.float32)
          else:

            if f'{scope}/s_w' in self._inits:
              s_w = tf.Variable(
                initial_value=self._inits[f'{scope}/s_w'].reshape((1, input_shape[1], input_shape[2], 1, self._num_neurons)),
                dtype=tf.float32)
            else:
              s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1, self._num_neurons),
                                dtype=tf.float32)
            if f'{scope}/d_w' in self._inits:
              d_w = tf.Variable(initial_value=self._inits[f'{scope}/d_w'].reshape(1, input_shape[-1], self._num_neurons),
                                dtype=tf.float32)
            else:
              d_w = tf.Variable(initial_value=np.random.randn(1, input_shape[-1], self._num_neurons),
                                dtype=tf.float32)
            if f'{scope}/bias' in self._inits:
              bias = tf.Variable(initial_value=self._inits[f'{scope}/bias'].reshape(1, self._num_neurons), dtype=tf.float32)
            else:
              bias = tf.Variable(initial_value=np.zeros((1, self._num_neurons)), dtype=tf.float32)

          tf.add_to_collection(f'{scope}/s_w', s_w)
          out = s_w * tf.expand_dims(self._input_ph, axis=-1)

          tf.add_to_collection(f'{scope}/d_w', d_w)
          out = tf.reduce_sum(out, axis=[1, 2])
          out = out * d_w

          tf.add_to_collection(f'{scope}/bias', bias)
          preds = tf.reduce_sum(out, axis=1) + bias

          self._predictions = tf.concat(preds, -1)
        elif self._map_type == 'separable_legacy':
          input_shape = self._input_ph.shape
          preds = []
          for n in range(self._num_neurons):
            with tf.variable_scope('N_{}'.format(n)):
              if self._inits is None:
                s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1), dtype=tf.float32)
                d_w = tf.Variable(initial_value=np.random.randn(1, 1, input_shape[-1], 1), dtype=tf.float32)
                bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)
              else:
                if 's_w' in self._inits:
                  s_w = tf.Variable(initial_value=self._inits['s_w'][n].reshape((1, input_shape[1], input_shape[2], 1)),
                                    dtype=tf.float32)
                else:
                  s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1),
                                    dtype=tf.float32)
                if 'd_w' in self._inits:
                  d_w = tf.Variable(initial_value=self._inits['d_w'][n].reshape(1, 1, input_shape[-1], 1),
                                    dtype=tf.float32)
                else:
                  d_w = tf.Variable(initial_value=np.random.randn(1, 1, input_shape[-1], 1), dtype=tf.float32)
                if 'bias' in self._inits:
                  bias = tf.Variable(initial_value=self._inits['bias'][n].reshape(1, 1, 1, 1), dtype=tf.float32)
                else:
                  bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)

              tf.add_to_collection(f'{scope}/s_w', s_w)
              out = s_w * self._input_ph

              tf.add_to_collection(f'{scope}/d_w', d_w)
              out = tf.reduce_sum(out, axis=[1, 2], keepdims=True)
              out = tf.nn.conv2d(out, d_w, [1, 1, 1, 1], 'SAME')

              tf.add_to_collection(f'{scope}/bias', bias)
              preds.append(tf.squeeze(out, axis=[1, 2]) + bias)

          self._predictions = tf.concat(preds, -1)
        elif self._map_type == 'linreg':
          # For L1-Regression
          tmp = tf.layers.flatten(self._input_ph)
          self._predictions = tf.layers.dense(tmp, self._num_neurons)
          weights = tf.global_variables()
          assert weights[1].shape == self._num_neurons
          tf.add_to_collection(f'{scope}/w', weights[0])
          tf.add_to_collection(f'{scope}/bias', weights[1])

  def _make_loss(self):
    """
    Makes the loss computational graph
    :return:
    """
    with self._graph.as_default():
      with tf.variable_scope('loss'):
        self.l2_error = tf.norm(self._predictions - self.target_ph, ord=2)
        # For L1-Regression
        if self._map_type == 'linreg':
          self._w = tf.get_collection('w')
          self._biases = tf.get_collection('bias')
          self.reg_loss = tf.reduce_sum(
            [tf.reduce_sum(tf.abs(t)) for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
          self.total_loss = self.l2_error + self._ld * self.reg_loss

        elif self._map_type == 'separable_legacy':
          # For separable mapping
          self._s_vars = tf.get_collection(f'{self._scope}/s_w')
          self._d_vars = tf.get_collection(f'{self._scope}/d_w')
          self._biases = tf.get_collection(f'{self._scope}/bias')

          # Laplacian loss
          laplace_filter = tf.constant(np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)),
                                       dtype=tf.float32)
          laplace_loss = tf.reduce_sum(
            [tf.norm(tf.nn.conv2d(t, laplace_filter, [1, 1, 1, 1], 'SAME')) for t in self._s_vars])
          l2_loss = tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self._s_vars])
          self.reg_loss = self._ls * (l2_loss + laplace_loss) + \
                          self._ld * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self._d_vars])

          self.total_loss = self.l2_error + self.reg_loss

        elif self._map_type == 'separable':
          # For separable mapping
          self._s_vars = tf.get_collection(f'{self._scope}/s_w')[0]
          self._d_vars = tf.get_collection(f'{self._scope}/d_w')[0]
          self._biases = tf.get_collection(f'{self._scope}/bias')[0]

          # Laplacian loss
          laplace_filter = tf.constant(npa([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)),
                                       dtype=tf.float32)
          laplace_loss = self._l2_loss(tf.nn.conv2d(tf.squeeze(tf.transpose(self._s_vars, perm=[4, 1, 2, 3, 0]),
                                                               axis=4), laplace_filter, [1, 1, 1, 1], 'SAME'))

          l2_loss_s = self._l2_loss(tf.transpose(self._s_vars))
          if self.use_l1loss:
            loss_d = self._l1_loss(tf.transpose(self._d_vars))
          else:
            loss_d = self._l2_loss(tf.transpose(self._d_vars))
          self.reg_loss = self._ls * laplace_loss + \
                          self._ld * (l2_loss_s + loss_d)

        if not self.multimode:
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
    print('Opening file to write to...')
    with h5py.File(save_path, 'w') as h5file:
      if 'separable' in self._map_type:
        h5file.create_dataset('s_w', data=self._sess.run(self._s_vars))
        h5file.create_dataset('d_w', data=self._sess.run(self._d_vars))
        h5file.create_dataset('bias', data=self._sess.run(self._biases))
      else:
        h5file.create_dataset('w', data=self._sess.run(self._w))
        h5file.create_dataset('bias', data=self._sess.run(self._biases))

    print('Finished saving.')

  def get_weights(self):
    if 'separable' in self._map_type:
      return {'s_w': self._sess.run(self._s_vars),
              'd_w': self._sess.run(self._d_vars),
              'bias': self._sess.run(self._biases)}
    else:
      return {'w': self._sess.run(self._w),
              'bias': self._sess.run(self._biases)}

  def _init_mapper(self, X):
    """
    Initializes the mapping function graph
    :param X: input data
    :return:
    """
    with self._graph.as_default():
      self._input_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
      self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._num_neurons])
      # Build the model graph
      self._make_separable_map()
      self._make_loss()
      self._is_initialized = True

      # initialize graph
      print('Initializing...')
      init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      if self._gpu_options is None:
        self._sess = tf.Session()
      else:
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=self._gpu_options))

      self._sess.run(init_op)
