import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops

from nsrec.nets import variable_scope_fn, end_points_collection_name, trunc_normal

default_scope_name = 'alexnet_v2'
image_height, image_width = 224, 224
variable_scope = variable_scope_fn(default_scope_name)


def cnn_layers(inputs, scope, end_points_collection, dropout_keep_prob=0.8, is_training=True):
  # Collect outputs for conv2d and max_pool2d.
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                      scope='conv1')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
    net = slim.conv2d(net, 192, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
    net = slim.conv2d(net, 384, [3, 3], scope='conv3')
    net = slim.conv2d(net, 384, [3, 3], scope='conv4')
    net = slim.conv2d(net, 256, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

  with slim.arg_scope([slim.conv2d],
                      weights_initializer=trunc_normal(0.005),
                      biases_initializer=tf.constant_initializer(0.1),
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                      scope='fc6')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout6')
    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout7')

  return net, end_points_collection


def fc_layers(net,
              scope,
              end_points_collection,
              num_classes=1000,
              is_training=True,
              dropout_keep_prob=0.5,
              spatial_squeeze=True,
              name_prefix=None):
  full_scope_name = lambda scope_name: scope_name if name_prefix is None else '%s_%s' % (name_prefix, scope_name)
  # Use conv2d instead of fully_connected layers.
  with slim.arg_scope([slim.conv2d],
                      weights_initializer=trunc_normal(0.005),
                      biases_initializer=tf.constant_initializer(0.1),
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(net, num_classes, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.zeros_initializer(),
                      scope=full_scope_name('fc8'))

  if spatial_squeeze:
    net = tf.squeeze(net, [1, 2], name=full_scope_name('fc8/squeezed'))
    ops.add_to_collection(end_points_collection, net)
  return net, end_points_collection