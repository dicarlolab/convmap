from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py

np.random.seed(123)
npa = np.array


# TF implementation of RF limited Regression


class Decoder(object):
  def __init__(self, graph=None, num_neurons=65, batch_size=50, init_lr=0.01,
               ls=0.05, ld=0.1, tol=1e-2, max_epochs=10, inits=None,
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
    self._inits = inits
    self._is_initialized = False
    self._log_rate = log_rate
    self._decay_rate = decay_rate
    self._gpu_options = gpu_options

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
    for start_idx in range(0, input_len, batchsize):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
      else:
        excerpt = slice(start_idx, start_idx + batchsize)
      if targets is None:
        yield inputs[excerpt]
      else:
        yield inputs[excerpt], targets[excerpt]

  def _make_separable_map(self):
    """
    Makes the mapping function computational graph
    :return:
    """
    with self._graph.as_default():
      with tf.variable_scope('decoder'):
        input_shape = self._input_ph.shape
        if self._inits is None:
          s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1, self._num_neurons),
                            dtype=tf.float32)
          d_w = tf.Variable(initial_value=np.random.randn(1, input_shape[-1], self._num_neurons), dtype=tf.float32)
          bias = tf.Variable(initial_value=np.zeros((1, self._num_neurons)), dtype=tf.float32)
        else:
          if 's_w' in self._inits:
            s_w = tf.Variable(
              initial_value=self._inits['s_w'].reshape((1, input_shape[1], input_shape[2], 1, self._num_neurons)),
              dtype=tf.float32)
          else:
            s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1, self._num_neurons),
                              dtype=tf.float32)
          if 'd_w' in self._inits:
            d_w = tf.Variable(initial_value=self._inits['d_w'].reshape(1, input_shape[-1], self._num_neurons),
                              dtype=tf.float32)
          else:
            d_w = tf.Variable(initial_value=np.random.randn(1, input_shape[-1], self._num_neurons),
                              dtype=tf.float32)
          if 'bias' in self._inits:
            bias = tf.Variable(initial_value=self._inits['bias'].reshape(1, self._num_neurons), dtype=tf.float32)
          else:
            bias = tf.Variable(initial_value=np.zeros((1, self._num_neurons)), dtype=tf.float32)

        tf.add_to_collection('s_w', s_w)
        out = s_w * tf.expand_dims(self._input_ph, axis=-1)

        tf.add_to_collection('d_w', d_w)
        out = tf.reduce_sum(out, axis=[1, 2])
        out = out * d_w

        tf.add_to_collection('bias', bias)
        preds = tf.reduce_sum(out, axis=1) + bias
        self._predictions = tf.concat(preds, -1)


  @staticmethod
  def _l2_loss(weights):
    return tf.reduce_mean([tf.reduce_sum(w ** 2) for w in weights])

  def _make_loss(self):
    """
    Makes the loss computational graph
    :return:
    """
    with self._graph.as_default():
      with tf.variable_scope('loss'):
        self._class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._predictions, labels=self.target_ph)
        # For separable mapping
        self._s_vars = tf.get_collection('s_w')[0]
        self._d_vars = tf.get_collection('d_w')[0]
        self._biases = tf.get_collection('bias')[0]

        # Laplacian loss
        laplace_filter = tf.constant(npa([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)),
                                     dtype=tf.float32)
        laplace_loss = self._l2_loss([tf.nn.conv2d(tf.squeeze(tf.transpose(self._s_vars, perm=[4, 1, 2, 3, 0]),
                                                              axis=4), laplace_filter, [1, 1, 1, 1], 'SAME')])

        l2_loss = self._l2_loss([self._s_vars, self._d_vars])
        self.reg_loss = self._ls * l2_loss + self._ld * laplace_loss

        self.total_loss = self._class_loss + self.reg_loss
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
      assert X.ndim == 4, 'Input matrix rank should be 4.'
      if self._is_initialized is False:
        self._init_mapper(X)

      for e in range(self._max_epochs):
        for counter, batch in enumerate(self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
          feed_dict = {self._input_ph: batch[0],
                       self.target_ph: batch[1],
                       self._lr_ph: self._lr}
          _, loss_value, reg_loss_value = self._sess.run([self.train_op, self._class_loss, self.reg_loss],
                                                         feed_dict=feed_dict)
        if e % self._log_rate == 0:
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
      for batch in self._iterate_minibatches(X, batchsize=self._batch_size, shuffle=False):
        feed_dict = {self._input_ph: batch}
        preds.append(np.squeeze(self._sess.run([self._predictions], feed_dict=feed_dict)))
      return np.concatenate(preds, axis=0)

  def save_weights(self, save_path):
    """
    Save weights to an hdf5 file
    :param save_path: save path
    :return:
    """
    print('Opening file to write to...')
    with h5py.File(save_path, 'w') as h5file:
      h5file.create_dataset('s_w', data=np.squeeze(self._sess.run(self._s_vars)))
      h5file.create_dataset('d_w', data=np.squeeze(self._sess.run(self._d_vars)))
      h5file.create_dataset('bias', data=np.squeeze(self._sess.run(self._biases)))
    print('Finished saving.')

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

  def close(self):
    """
    Closes occupied resources
    :return:
    """
    tf.reset_default_graph()
    self._sess.close()
