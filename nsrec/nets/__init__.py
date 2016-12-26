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

