import tensorflow as tf
from tensorflow.contrib import slim

from nsrec.nets import variable_scope_fn, end_points_collection_name

default_scope_name = 'lenet'
image_height, image_width = 28, 28
variable_scope = variable_scope_fn(default_scope_name)


def cnn_layers(inputs, scope, end_points_collection):
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)

  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  return net, end_points


def fc_layers(net,
              scope,
              end_points_collection,
              num_classes=10,
              is_training=True,
              dropout_keep_prob=0.5,
              name_prefix=None):

  def full_scope_name(scope_name):
    return scope_name if name_prefix is None else '%s_%s' % (name_prefix, scope_name)

  with slim.arg_scope([slim.fully_connected, slim.dropout],
                      outputs_collections=[end_points_collection]):
    net = slim.fully_connected(net, 1024, scope=full_scope_name('fc3'))
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope=full_scope_name('dropout3'))
    net = slim.fully_connected(net, num_classes, activation_fn=None,
                               scope=full_scope_name('fc4'))

  end_points = slim.utils.convert_collection_to_dict(end_points_collection)

  return net, end_points