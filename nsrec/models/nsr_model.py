from functools import reduce

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import rnn
from nsrec.models.model_helper import all_model_variables_data, vars_assign_ops, softmax_accuracy, global_step_variable, \
  gray_scale, stack_output_ops
from nsrec.utils.np_ops import correct_count


class CNNNSRTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.is_training = True
    self.max_number_length = self.config.max_number_length
    self.batch_size = config.batch_size

    self.total_loss = None
    self.global_step = None
    self.length_output = None
    self.numbers_output = None

    self.data_batches = None
    self.length_label_batches = None
    self.numbers_label_batches = []

  def _setup_input(self):
    config = self.config
    self.data_batches, self.length_label_batches, self.numbers_label_batches = \
        nsr_data_batches(config.batch_size, config.data_file_path, config.max_number_length, config.size,
                         self.is_training, self.config.channels, self.config.num_preprocess_threads)

  def _setup_net(self):
    self.length_output, self.numbers_output = nsr_net(
      self.cnn_net, self.data_batches, self.config.max_number_length, self.is_training)

  def _setup_loss(self):
    number_losses = []
    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.length_output, labels=self.length_label_batches)
      # tf.log will cause NaN issue
      # length_loss = tf.log(tf.reduce_mean(length_loss), 'length_loss')
      length_loss = tf.reduce_mean(length_loss, name='length_loss')
      self.total_loss = length_loss

      for i in range(self.max_number_length):
        number_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.numbers_output[i], labels=self.numbers_label_batches[i])
        # tf.log will cause NaN issue
        # number_loss = tf.log(tf.reduce_mean(number_loss), 'number%s_loss' % (i + 1))
        number_loss = tf.reduce_mean(number_loss, name='number%s_loss' % (i + 1))
        number_losses.append(number_loss)
        self.total_loss = self.total_loss + number_loss

    tf.summary.scalar("loss/length_loss", length_loss)
    for i in range(self.max_number_length):
      tf.summary.scalar("loss/number%s_loss" % (i + 1), number_losses[i])
    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  def _setup_accuracy(self, op_name='accuracy/train'):
    softmax_accuracy(self.length_output, self.length_label_batches, 'accuracy/train/length')
    for i in range(self.max_number_length):
      softmax_accuracy(self.numbers_output[i], self.numbers_label_batches[i], 'accuracy/train/number%s' % (i + 1))

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_loss()
    self._setup_accuracy()
    self._setup_global_step()


class CNNNSREvalModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.max_number_length = self.config.max_number_length
    self.batch_size = config.batch_size
    self.is_training = False

    self.length_output = None
    self.numbers_output = None

    self.data_batches = None
    self.length_label_batches = None
    self.numbers_label_batches = []

    self.length_label_batches_pd = None
    self.numbers_label_batches_pd = None

  def _setup_input(self):
    config = self.config
    self.data_batches, self.length_label_batches, self.numbers_label_batches = \
        nsr_data_batches(config.batch_size, config.data_file_path, config.max_number_length, config.size,
                         self.is_training, self.config.channels, self.config.num_preprocess_threads)

  def _setup_net(self):
    self.length_output, self.numbers_output = nsr_net(
      self.cnn_net, self.data_batches, self.config.max_number_length, self.is_training)
    self.length_label_batches_pd = tf.nn.softmax(self.length_output)
    self.numbers_label_batches_pd = [None] * self.max_number_length
    for i in range(self.max_number_length):
      self.numbers_label_batches_pd[i] = tf.nn.softmax(self.numbers_output[i])

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_global_step()

  def correct_count(self, sess):
    calculated_values = sess.run({
      'length_label_batches': self.length_label_batches,
      'numbers_label_batches': self.numbers_label_batches,
      'length_label_batches_pd': self.length_label_batches_pd,
      'numbers_label_batches_pd': self.numbers_label_batches_pd
    })
    length_label_batches, numbers_label_batches, length_label_batches_pd, numbers_label_batches_pd = \
        calculated_values['length_label_batches'], calculated_values['numbers_label_batches'], \
        calculated_values['length_label_batches_pd'], calculated_values['numbers_label_batches_pd']

    return correct_count(length_label_batches, numbers_label_batches,
                         length_label_batches_pd, numbers_label_batches_pd)


class CNNNSRInferenceModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.max_number_length = config.max_number_length
    self.is_training = False

    self.inputs = None
    self.data_batches = None

    self.length_pb = None
    self.numbers_pb = []

    self.global_step = None

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self):
    self.length_output, self.numbers_output = nsr_net(
      self.cnn_net, self.data_batches, self.config.max_number_length, self.is_training)
    self.length_pb = tf.nn.softmax(self.length_output)
    for i in range(self.max_number_length):
      self.numbers_pb.append(tf.nn.softmax(self.numbers_output[i]))

  def build(self):
    self._setup_input()
    self._setup_net()

  def infer(self, sess, data):
    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    length_pb, numbers_pb = sess.run(
      [self.length_pb, self.numbers_pb],
      feed_dict={self.inputs: input_data})
    length = np.argmax(length_pb, axis=1)
    numbers = np.argmax(numbers_pb, axis=2)

    labels = []
    length_prob = lambda i: length_pb[i][length[i]]
    number_prob = lambda i, j: numbers_pb[j][i][numbers[j][i]]

    for i in range(len(length)):
      label = ''.join([str(n) for n in numbers[:length[i] + 1, i]])
      probabilities = [length_prob(i)] + [number_prob(i, j) for j in range(length[i] + 1)]
      reduced_prob = reduce(lambda x, y: x * y, probabilities)
      labels.append((label, probabilities, reduced_prob))
    return labels

  def vars(self, sess):
    return all_model_variables_data(sess)


class CNNNSRToExportModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.max_number_length = config.max_number_length
    self.is_training = False

    self.inputs = None
    self.data_batches = None

    self.output = None
    self.global_step = None

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self, saved_vars_dict):
    self.length_output, self.numbers_output = nsr_net(
      self.cnn_net, self.data_batches, self.config.max_number_length, self.is_training)

    assign_ops = vars_assign_ops(self._vars(), saved_vars_dict)
    with tf.control_dependencies(assign_ops):
      self.output = stack_output_ops(
        self.max_number_length, self.length_output, self.numbers_output, name='output')

  def build(self, saved_vars_dict):
    self._setup_input()
    self._setup_net(saved_vars_dict)


def nsr_net(cnn_net, data_batches, max_number_length, is_training):
  with cnn_net.variable_scope([data_batches]) as variable_scope:
    end_points_collection_name = cnn_net.end_points_collection_name(variable_scope)
    net, end_points_collection = cnn_net.cnn_layers(data_batches, variable_scope, end_points_collection_name)
    length_output, _ = cnn_net.fc_layers(
      net, variable_scope, end_points_collection, dropout_keep_prob=0.9,
      num_classes=max_number_length, is_training=is_training, name_prefix='length')
    numbers_output = []
    for i in range(max_number_length):
      number_output, _ = cnn_net.fc_layers(
        net, variable_scope, end_points_collection, dropout_keep_prob=0.9,
        is_training=is_training, num_classes=11, name_prefix='number%s' % (i + 1))
      numbers_output.append(number_output)
  return length_output, numbers_output


def nsr_data_batches(batch_size, data_file_path, max_number_length, size, is_training, channels, num_preprocess_threads):
  with ops.name_scope(None, 'Input') as sc:
    data_batches, length_label_batches, combined_numbers_label_batches = \
      inputs.batches(data_file_path, max_number_length, batch_size, size,
                     is_training=is_training, channels=channels,
                     num_preprocess_threads=num_preprocess_threads)
    numbers_label_batches = []
    for i in range(max_number_length):
      numbers_label_batches.append(combined_numbers_label_batches[:, i, :])
    return data_batches, length_label_batches, numbers_label_batches


class RNNTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.is_training = True
    self.max_number_length = self.config.max_number_length
    self.batch_size = config.batch_size
    self.embedding_size = 256

    self.total_loss = None
    self.global_step = None
    self.length_output = None
    self.numbers_output = None

    self.data_batches = None
    self.length_label_batches = None
    self.numbers_label_batches = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, _, numbers_label_batches = \
        inputs.batches(config.data_file_path, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training, channels=config.channels)
      self.numbers_label_batches = tf.transpose(numbers_label_batches, perm=[1, 0, 2])

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection = self.cnn_net.end_points_collection_name(variable_scope)
      net, _ = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection)
      net = slim.fully_connected(net, self.embedding_size, activation_fn=None, scope='fc0')
      net = rnn.rnn_layers(net, tf.arg_max(self.numbers_label_batches, dimension=2), self.embedding_size)
      net = tf.reshape(net, [-1, self.embedding_size])
      self.model_output = slim.fully_connected(net, 11, activation_fn=None, scope='fc4')

  def _setup_loss(self):
    numbers_label_batches = tf.reshape(self.numbers_label_batches, [-1, 11])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.model_output, labels=numbers_label_batches)

    batch_loss = tf.reduce_mean(losses)
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()

    # Add summaries.
    tf.summary.scalar("loss/batch_loss", batch_loss)
    tf.summary.scalar("loss/total_loss", total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    self.total_loss = total_loss

  def _setup_accuracy(self, op_name='accuracy/train'):
    numbers_label_batches = tf.reshape(self.numbers_label_batches, [-1, 11])
    softmax_accuracy(self.model_output, numbers_label_batches, op_name)

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_loss()
    self._setup_accuracy()
    self._setup_global_step()


class RNNEvalModel:
  pass
