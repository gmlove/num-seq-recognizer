from tensorflow.contrib import slim

from nsrec.nets import variable_scope_fn, end_points_collection_name

default_scope_name = 'lenet_v1'
image_height, image_width = 64, 64
variable_scope = variable_scope_fn(default_scope_name)

'''
best accuracy: 0.71, data: 4.2M
'''

def cnn_layers(inputs, scope, end_points_collection, dropout_keep_prob=0.8, is_training=True):
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                      outputs_collections=[end_points_collection]):
    net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 64, [5, 5], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = slim.conv2d(net, 64, [5, 5], scope='conv4')
    net = slim.conv2d(net, 64, [5, 5], scope='conv5')
    net = slim.conv2d(net, 64, [5, 5], scope='conv6')
    net = slim.conv2d(net, 64, [5, 5], scope='conv7')
    net = slim.flatten(net)

    net = slim.fully_connected(net, 128, scope='fc3')

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
    '''
    with droupout accuracy: 0.68, data: 4.2M
    without droupout accuracy: 0.71, data: 4.2M
    '''
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope=full_scope_name('dropout3'))
    net = slim.fully_connected(net, num_classes, activation_fn=None,
                               scope=full_scope_name('fc4'))

  return net, end_points_collection