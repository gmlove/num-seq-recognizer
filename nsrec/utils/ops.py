import tensorflow as tf
from tensorflow.python.framework import ops


def all_model_variables_data(sess):
  coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
  model_vars = sess.run(coll)
  model_vars_dict = dict(zip([v.name for v in coll], model_vars))
  return model_vars_dict


def assign_vars(vars_dict, saved_vars_dict, prefix=''):
  assign_ops = []
  for name, input_var in saved_vars_dict.items():
    assign_ops.append(tf.assign(vars_dict[prefix + name], input_var))
  return assign_ops


def softmax_accuracy(logits, label_batches, scope_name):
  with ops.name_scope(None, scope_name) as sc:
    correct_prediction = tf.equal(
      tf.argmax(tf.nn.softmax(logits), 1),
      tf.argmax(label_batches, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar(scope_name, accuracy)

    return accuracy


def global_step_variable():
  return tf.Variable(initial_value=0, name="global_step", trainable=False,
                     collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])


def gray_scale(inputs):
  data_batches = tf.image.rgb_to_grayscale(inputs)
  shape = data_batches.get_shape().as_list()
  shape[-1] = 1
  data_batches.set_shape(shape)
  return data_batches


def stack_output(max_number_length, length_output, numbers_output, name='output'):
  length_pb = tf.nn.softmax(length_output)
  to_concat = [tf.reshape(length_pb, (max_number_length, ))]
  for i in range(max_number_length):
    to_concat.append(tf.reshape(tf.nn.softmax(numbers_output[i]), (11, )))
  return tf.concat(axis=0, values=to_concat, name=name)


def softmax_cross_entrophy_loss(logits, labels):
  with ops.name_scope(None, 'Loss') as sc:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    total_loss = loss

  tf.summary.scalar("loss/total_loss", total_loss)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  return total_loss