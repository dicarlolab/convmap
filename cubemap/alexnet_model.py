from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from collections import OrderedDict
from TF_model import TFModel
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


class Alexnet(TFModel):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.default_image_size = 224

    def arg_scope(self, weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)
                            ):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def model(self,
              inputs,
              num_classes,
              is_training=True,
              dropout_keep_prob=0.5,
              spatial_squeeze=True,
              scope='alexnet_v2'):
        """AlexNet version 2.
        Described in: http://arxiv.org/pdf/1404.5997v2.pdf
        Parameters from:
        github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
        layers-imagenet-1gpu.cfg
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224. To use in fully
              convolutional mode, set spatial_squeeze to false.
              The LRN layers have been removed and change the initializers from
              random_normal_initializer to xavier_initializer.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
        Returns:
          the last op containing the log predictions and end_points dict.
        """

        # with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points = OrderedDict()
        with tf.name_scope(scope, 'alexnet_v2', [inputs]) as sc:
            # with tf.name_scope(scope, 'alexnet_v2', [inputs]) as sc:
            end_points_collection = sc + 'end_points'
            # print('ENDPOINTS: {0}'.format(end_points_collection))
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID',
                                  scope='conv1')
                end_points['conv1'] = net
                # LRN layer not from slim
                net = tf.nn.local_response_normalization(net, depth_radius=5, alpha=0.0001, beta=0.75)
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
                end_points['pool1'] = net
                net = slim.conv2d(net, 256, [5, 5], scope='conv2')
                end_points['conv2'] = net
                net = tf.nn.local_response_normalization(net, depth_radius=5, alpha=0.0001, beta=0.75)
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                end_points['pool2'] = net
                net = slim.conv2d(net, 384, [3, 3], scope='conv3')
                end_points['conv3'] = net
                net = slim.conv2d(net, 384, [3, 3], scope='conv4')
                end_points['conv4'] = net
                net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                end_points['conv5'] = net
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
                end_points['pool5'] = net

                # Use conv2d instead of fully_connected layers.
                with slim.arg_scope([slim.conv2d],
                                    weights_initializer=trunc_normal(0.005),
                                    biases_initializer=tf.constant_initializer(0.1)):
                    net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                      scope='fc6')
                    end_points['fc6'] = net
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    end_points['fc7'] = net
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer(dtype=tf.float32),
                                      scope='fc8')
                    end_points['fc8'] = net

                # Convert end_points_collection into a end_point dict.
                # end_points = dict(tf.get_collection(end_points_collection))
                # end_points = dict((v.name,v) for v in tf.get_collection(end_points_collection))
                # print(end_points.keys())
                if spatial_squeeze:
                    net = tf.squeeze(net, name='fc8/squeezed')
                    end_points[sc + 'fc8'] = net

                return net, end_points

    def lr_generator(self, global_step, decay_steps):
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)
        return lr

    def optimizer(self, lr):
      opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
      return opt


def main(_):
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=tuple([None] + [224, 224, 3]))
    model = Alexnet()
    logits, endpoints = model.model(images_placeholder, 1001)
    print(endpoints.keys())
    print(endpoints['fc7'])


if __name__ == '__main__':
    tf.app.run()
