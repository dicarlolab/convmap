from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py

np.random.seed(123)


# TF implementation of RF limited Regression

class SeparableMap(object):
  def __init__(self, graph=None, num_neurons=65, batch_size=50, init_lr=0.01,
               ls=0.05, ld=0.1, tol=1e-2, max_epochs=10, map_type='linreg', init_rfs=None):
    self._ld = ld  # reg factor for depth conv
    self._ls = ls  # reg factor for spatial conv
    self._tol = tol
    self._batch_size = batch_size
    self._num_neurons = num_neurons
    self._lr = init_lr
    self._max_epochs = max_epochs
    self._map_type = map_type
    self._init_rfs = init_rfs

    tf.reset_default_graph()

    if graph is None:
      self._graph = tf.Graph()
    else:
      self._graph = graph

    with self._graph.as_default():
      self._lr_ph = tf.placeholder(dtype=tf.float32)
      self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

  def _iterate_minibatches(self, inputs, targets=None, batchsize=128, shuffle=False):
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
    with self._graph.as_default() as graph:
      with tf.variable_scope('mapping'):
        if self._map_type == 'separable':
          input_shape = self.input_ph.shape
          preds = []
          for n in range(self._num_neurons):
            with tf.variable_scope('N_{}'.format(n)):
              if self._init_rfs is None:
                s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1), dtype=tf.float32)
              else:
                assert self._init_rfs.shape == (self._num_neurons, input_shape[1], input_shape[2]), \
                  'Filter initialization matrix should be ({},{},{})'.format(self._num_neurons, input_shape[1],
                                                                             input_shape[2])
                s_w = tf.Variable(initial_value=self._init_rfs[n].reshape((1, input_shape[1], input_shape[2], 1)),
                                  dtype=tf.float32)
              tf.add_to_collection('s_w', s_w)
              out = self.input_ph * s_w
              d_w = tf.Variable(initial_value=np.random.randn(1, 1, out.shape[-1], 1), dtype=tf.float32)
              tf.add_to_collection('d_w', d_w)
              out = tf.nn.conv2d(out, d_w, [1, 1, 1, 1], 'SAME')
              bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)
              tf.add_to_collection('bias', bias)
              preds.append(tf.reduce_sum(out, axis=[1, 2]) + bias)
          self._predictions = tf.concat(preds, -1)
        elif self._map_type == 'linreg':
          # For L1-Regression
          tmp = tf.layers.flatten(self.input_ph)
          self._predictions = tf.layers.dense(tmp, self._num_neurons)

  def _make_loss(self):
    with self._graph.as_default():
      with tf.variable_scope('loss'):
        self.l2_error = tf.norm(self._predictions - self.target_ph,
                                ord=2)  # tf.reduce_sum(tf.pow(self._predictions-self.target_ph, 2))/(2*self.batch_size) #

        # For L1-Regression
        if self._map_type == 'linreg':
          self.reg_loss = tf.reduce_sum(
            [tf.reduce_sum(tf.abs(t)) for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
          self.total_loss = self.l2_error + self.reg_loss

        elif self._map_type == 'separable':
          # For separable mapping
          self.s_vars = tf.get_collection('s_w')
          self.d_vars = tf.get_collection('d_w')
          self.biases = tf.get_collection('bias')

          # L1 reg
          # self.reg_loss = self.ls * tf.reduce_sum([tf.reduce_sum(tf.abs(t)) for t in self.s_vars]) + self.ld * tf.reduce_sum([tf.reduce_sum(tf.abs(t)) for t in self.d_vars])
          # L2 reg
          # self.reg_loss = self.ls * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.s_vars]) + self.ld * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.d_vars])
          #                 self.total_loss = self.l2_error + self.reg_loss

          # Laplacian loss
          laplace_filter = tf.constant(np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)), dtype=tf.float32)
          laplace_loss = tf.reduce_sum(
            [tf.norm(tf.nn.conv2d(t, laplace_filter, [1, 1, 1, 1], 'SAME')) for t in self.s_vars])
          l2_loss = tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.s_vars])
          self.reg_loss = self._ls * (l2_loss + laplace_loss) + \
                          self._ld * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.d_vars])

          self.total_loss = self.l2_error + self.reg_loss
        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                          global_step=tf.contrib.framework.get_or_create_global_step())

  def fit(self, X, Y):
    with self._graph.as_default():
      if self._map_type == 'linreg':
        assert X.ndim == 2, 'Input matrix rank should be 2.'
      else:
        assert X.ndim == 4, 'Input matrix rank should be 4.'
      self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
      self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, Y.shape[-1]])
      # Build the model graph
      self._make_separable_map()
      self._make_loss()

      # initialize graph
      print('Initializing...')
      self.sess = tf.Session()
      init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      self.sess.run(init_op)
      for e in range(self._max_epochs):
        for counter, batch in enumerate(self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
          feed_dict = {self.input_ph: batch[0],
                       self.target_ph: batch[1],
                       self._lr_ph: self._lr}
          _, loss_value, reg_loss_value = self.sess.run([self.train_op, self.l2_error, self.reg_loss],
                                                        feed_dict=feed_dict)
        if e % 100 == 0:
          print('Epoch: %d, Err Loss: %.2f, Reg Loss: %.2f' % (e + 1, loss_value, reg_loss_value))
        if e % 200 == 0 and e != 0:
          self._lr /= 10.
        if loss_value < self._tol:
          print('Converged.')
          break

  def predict(self, X):
    preds = []
    with self._graph.as_default():
      for batch in self._iterate_minibatches(X, batchsize=self._batch_size, shuffle=False):
        feed_dict = {self.input_ph: batch}
        preds.append(np.squeeze(self.sess.run([self._predictions], feed_dict=feed_dict)))
      return np.concatenate(preds, axis=0)

  def save_weights(self, save_path):

    print('Opening file to write to...')
    with h5py.File(save_path, 'w') as h5file:
      h5file.create_dataset('s_w', data=np.squeeze(self.sess.run(self.s_vars)))
      h5file.create_dataset('s_d', data=np.squeeze(self.sess.run(self.d_vars)))
      h5file.create_dataset('bias', data=np.squeeze(self.sess.run(self.biases)))
    print('Finished saving.')

  def close(self):
    tf.reset_default_graph()
    self.sess.close()



