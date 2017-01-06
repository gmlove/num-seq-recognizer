from tensorflow.contrib import slim

from nsrec.nets import variable_scope_fn, end_points_collection_name

default_scope_name = 'iclr_mnr'
image_height, image_width = 64, 64
variable_scope = variable_scope_fn(default_scope_name)


def cnn_layers(inputs, scope, end_points_collection):
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(inputs, 48, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1', padding='SAME')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')
    net = slim.conv2d(net, 128, [5, 5], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3', padding='SAME')
    net = slim.conv2d(net, 160, [5, 5], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')
    net = slim.conv2d(net, 192, [5, 5], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool5', padding='SAME')
    net = slim.conv2d(net, 192, [5, 5], scope='conv6')
    net = slim.max_pool2d(net, [2, 2], scope='pool6', padding='SAME')
    net = slim.conv2d(net, 192, [5, 5], scope='conv7')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool7', padding='SAME')
    net = slim.flatten(net)

    net = slim.fully_connected(net, 3072, scope='fc8')

  return net, end_points_collection


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
    net = slim.fully_connected(net, num_classes, activation_fn=None,
                               scope=full_scope_name('fc9'))

  return net, end_points_collection