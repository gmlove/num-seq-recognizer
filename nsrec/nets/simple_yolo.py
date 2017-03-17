import tensorflow as tf
from tensorflow.contrib import slim
from nsrec.nets import variable_scope_fn, end_points_collection_name
from nsrec.utils.ops import leaky_relu

default_scope_name = 'simple_yolo'
image_height, image_width = 104, 104
variable_scope = variable_scope_fn(default_scope_name)


def _reorg(net, stride):
  return tf.extract_image_patches(
    net, [1, stride, stride, 1], [1, stride, stride, 1], [1, 1, 1, 1], 'VALID')


def cnn_layers(inputs, scope, end_points_collection, dropout_keep_prob=0.8, is_training=True):
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                      outputs_collections=[end_points_collection]):
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training},
                        activation_fn=leaky_relu):
      net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
      net = slim.conv2d(net, 128, [3, 3], scope='conv3')
      net = slim.conv2d(net, 64, [1, 1], scope='conv4')
      box_net = net = slim.conv2d(net, 128, [3, 3], scope='conv5')

      net = slim.max_pool2d(net, [2, 2], 2, scope='pool5')
      net = slim.conv2d(net, 256, [3, 3], scope='conv6')
      net = slim.conv2d(net, 128, [1, 1], scope='conv7')
      net = slim.conv2d(net, 256, [3, 3], scope='conv8')

      box_net = _reorg(box_net, 2)
      net = tf.concat([box_net, net], 3)
      net = slim.conv2d(net, 256, [3, 3], scope='conv9')
      net = slim.conv2d(net, 75, [1, 1], activation_fn=None, scope='conv10')

  return net, end_points_collection
