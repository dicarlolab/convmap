from __future__ import print_function
import tensorflow as tf
import re
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

FLAGS = tf.app.flags.FLAGS


class TFModel(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.MOVING_AVERAGE_DECAY = 0.9999

        # Constants dictating the learning rate schedule.
        self.RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
        self.RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
        self.RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

    def arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def model(self,
              inputs,
              num_classes,
              dropout_keep_prob=0.8,
              is_training=False,
              spatial_squeeze=True,
              scope='model'):
        raise NotImplementedError()

    def inference(self,
                  images,
                  num_classes,
                  is_training=False,
                  spatial_squeeze=True,
                  scope=None):

        with slim.arg_scope(self.arg_scope()):
            logits, endpoints = self.model(images,
                                           num_classes,
                                           is_training=is_training,
                                           spatial_squeeze=spatial_squeeze,
                                           scope=scope)

        return logits

    def loss(self,
             logits,
             labels,
             batch_size=None):
        print('Using default loss (softmax-Xentropy)...')
        if not batch_size:
            batch_size = self.batch_size

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat([indices, sparse_labels], 1)
        num_classes = logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
                                          [batch_size, num_classes],
                                          1.0, 0.0)
        slim.losses.softmax_cross_entropy(logits,
                                          dense_labels,
                                          label_smoothing=0.1,
                                          weights=1.0)

    def lr_generator(self, global_step=None, decay_steps=None):
        print('Using default learning rate scheduler (CONSTANT)...')
        lr = ops.convert_to_tensor(FLAGS.initial_learning_rate, name="learning_rate")
        return lr

    def optimizer(self, lr):
        print('Using default optimizer (RMSPROP)...')
        opt = tf.train.RMSPropOptimizer(lr, self.RMSPROP_DECAY,
                                        momentum=self.RMSPROP_MOMENTUM,
                                        epsilon=self.RMSPROP_EPSILON)
        return opt

    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        Args:
          x: Tensor
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _activation_summaries(self, endpoints):
        with tf.name_scope('summaries'):
            for act in endpoints.values():
                self._activation_summary(act)
