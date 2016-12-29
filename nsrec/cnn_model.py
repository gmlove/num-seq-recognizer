import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import lenet, alexnet, inception_v3
from nsrec.np_ops import correct_count


class CNNModelConfig(object):

  def __init__(self, **kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.data_dir_path = os.path.join(current_dir, '../data/train')
    self.metadata_file_path = os.path.join(self.data_dir_path, 'digitStruct.mat')
    self.max_number_length = 5
    self.batch_size = 64
    self.create_metadata_handler_fn = inputs.create_pickle_metadata_handler
    self.net_type = "lenet"
    self.force_size = None

    for attr in ['data_dir_path', 'metadata_file_path', 'max_number_length',
                 'batch_size', 'force_size', 'create_metadata_handler_fn', 'net_type']:
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))

  @property
  def size(self):
    if self.force_size is not None:
      return self.force_size
    return [self.cnn_net.image_height, self.cnn_net.image_width]

  @property
  def cnn_net(self):
    if self.net_type == 'alexnet':
      return alexnet
    elif self.net_type == 'inception_v3':
      return inception_v3
    else:
      return lenet

class CNNGeneralModelConfig(CNNModelConfig):

  def __init__(self, **kwargs):
    super(CNNGeneralModelConfig, self).__init__(**kwargs)
    self.num_classes = self.max_number_length

    for attr in ['num_classes']:
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))


class CNNGeneralModelBase:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.model_output = None
    self.is_training = True

  def _build_base_net(self):
    self._pre_build(self.config)

    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection_name = self.cnn_net.end_points_collection_name(variable_scope)
      net, end_points_collection = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection_name)
      self.model_output, _ = self.cnn_net.fc_layers(
        net, variable_scope, end_points_collection,
        num_classes=self.config.num_classes, is_training=self.is_training, name_prefix='length')

  def _pre_build(self):
    pass


class CNNMnistTrainModel(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNMnistTrainModel, self).__init__(config)

    self.label_batches = None
    self.total_loss = None
    self.global_step = None

  def _pre_build(self, config):
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches = \
        inputs.mnist_batches(config.batch_size, config.size, is_training=self.is_training)

  def build(self):
    self._build_base_net()

    with ops.name_scope(None, 'Loss') as sc:
      loss = tf.nn.softmax_cross_entropy_with_logits(self.model_output, self.label_batches)
      loss = tf.reduce_mean(loss)
      self.total_loss = loss

    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    self.setup_global_step()

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step


class CNNLengthTrainModel(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNLengthTrainModel, self).__init__(config)

    self.total_loss = None
    self.global_step = None

  def _pre_build(self, config):
    with ops.name_scope(None, 'Input') as sc:
      metadata_handler = config.create_metadata_handler_fn(
        config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, self.length_label_batches, self.numbers_label_batches = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training)

  def build(self):
    self._build_base_net()

    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(self.model_output, self.length_label_batches)
      length_loss = tf.log(tf.reduce_mean(length_loss), 'length_loss')
      self.total_loss = length_loss

    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    self.setup_global_step()

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step


class CNNModelBase:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.length_output = None
    self.numbers_output = None
    self.is_training = True

  def _build_base_net(self):
    self._pre_build(self.config)

    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection_name = self.cnn_net.end_points_collection_name(variable_scope)
      net, end_points_collection = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection_name)
      self.length_output, _ = self.cnn_net.fc_layers(
        net, variable_scope, end_points_collection,
        num_classes=self.config.max_number_length, is_training=self.is_training, name_prefix='length')
      self.numbers_output = []
      for i in range(self.config.max_number_length):
        number_output, _ = self.cnn_net.fc_layers(
          net, variable_scope, end_points_collection,
          is_training=self.is_training, num_classes=10, name_prefix='number%s' % (i + 1))
        self.numbers_output.append(number_output)

  def _pre_build(self):
    pass

class CNNTrainModel(CNNModelBase):

  def __init__(self, config):
    super(CNNTrainModel, self).__init__(config)

    self.total_loss = None
    self.global_step = None

  def _pre_build(self, config):
    with ops.name_scope(None, 'Input') as sc:
      metadata_handler = config.create_metadata_handler_fn(
        config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, self.length_label_batches, self.numbers_label_batches = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training)

  def build(self):
    self._build_base_net()

    number_losses = []
    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(self.length_output, self.length_label_batches)
      length_loss = tf.log(tf.reduce_sum(length_loss), 'length_loss')
      self.total_loss = length_loss

      for i in range(self.config.max_number_length):
        number_loss = tf.nn.softmax_cross_entropy_with_logits(self.numbers_output[i], self.numbers_label_batches[:,i,:])
        number_loss = tf.log(tf.reduce_sum(number_loss), 'number%s_loss' % (i + 1))
        number_losses.append(number_loss)
        self.total_loss = self.total_loss + number_loss

    tf.summary.scalar("loss/length_loss", length_loss)
    for i in range(self.config.max_number_length):
      tf.summary.scalar("loss/number%s_loss" % (i + 1), number_losses[i])
    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    self.setup_global_step()

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step


class CNNEvalModel(CNNTrainModel):

  def __init__(self, config):
    super(CNNEvalModel, self).__init__(config)
    self.is_training = False

  def build(self):
    super(CNNEvalModel, self).build()

    with ops.name_scope(None, 'EvalOutput') as sc:
      self.length_label_batches_pd = tf.nn.softmax(self.length_output, name="length_output/softmax")
      self.numbers_label_batches_pd = []
      for i in range(self.config.max_number_length):
        number_prob_distribution = tf.nn.softmax(self.numbers_output[i], name="number%s_output/softmax" % i)
        self.numbers_label_batches_pd.append(number_prob_distribution)

  def correct_count(self, sess):
    calculated_values = sess.run({
      'length_label_batches': self.length_label_batches,
      'numbers_label_batches': self.numbers_label_batches,
      'length_label_batches_pd': self.length_label_batches_pd,
      'numbers_label_batches_pd': self.numbers_label_batches_pd
    })
    length_label_batches, numbers_label_batches, \
    length_label_batches_pd, numbers_label_batches_pd = \
      calculated_values['length_label_batches'], calculated_values['numbers_label_batches'], \
      calculated_values['length_label_batches_pd'], calculated_values['numbers_label_batches_pd']

    # numbers_label_batches.shape is (batch_size, max_number_length, 10)
    # numbers_label_batches_pd.shape is (max_number_length, batch_size, 10)
    # transform numbers_label_batches_pd to be the same as numbers_label_batches
    normalized_numbers_label_batches_pd = []
    for i in range(self.config.batch_size):
      normalized_numbers_label_batches_pd.append([])
      for j in range(self.config.max_number_length):
        normalized_numbers_label_batches_pd[i].append(numbers_label_batches_pd[j][i])

    return correct_count(length_label_batches, numbers_label_batches,
                         length_label_batches_pd, normalized_numbers_label_batches_pd)


class CNNPredictModel(CNNModelBase):
  pass
