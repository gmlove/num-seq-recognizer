import tensorflow as tf


def variable_scope_fn(default_scope_name):
  def variable_scope(values=None, name_or_scope=None, **kwargs):
    values = values if isinstance(values, (tuple, list)) else [values]
    kwargs.update({'values': values, 'default_name': default_scope_name})
    return tf.variable_scope(name_or_scope, **kwargs)
  return variable_scope


def end_points_collection_name(variable_scope):
  end_points_collection = variable_scope.original_name_scope + '_end_points'
  return end_points_collection


def trunc_normal(stddev):
  return tf.truncated_normal_initializer(0.0, stddev)


def basic_net(cnn_net, data_batches, num_classes, is_training):
  with cnn_net.variable_scope([data_batches]) as variable_scope:
    end_points_collection_name = cnn_net.end_points_collection_name(variable_scope)
  net, end_points_collection = cnn_net.cnn_layers(
    data_batches, variable_scope, end_points_collection_name)
  model_output, _ = cnn_net.fc_layers(
    net, variable_scope, end_points_collection,
    num_classes=num_classes, is_training=is_training, name_prefix='length')
  return model_output
