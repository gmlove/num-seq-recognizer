import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import lenet, alexnet


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
    else:
      return lenet


class CNNModelBase:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.length_output = None
    self.numbers_output = None

  def _build_base_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection = self.cnn_net.end_points_collection_name(variable_scope)
      net, _ = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection)
      self.length_output, _ = self.cnn_net.fc_layers(net, variable_scope, end_points_collection,
                                           num_classes=self.config.max_number_length,
                                           name_prefix='length')
      self.numbers_output = []
      for i in range(self.config.max_number_length):
        number_output, _ = self.cnn_net.fc_layers(net, variable_scope, end_points_collection,
                                             num_classes=10, name_prefix='number%s' % (i + 1))
        self.numbers_output.append(number_output)


class CNNTrainModel(CNNModelBase):

  def __init__(self, config):
    super(CNNTrainModel, self).__init__(config)

    with ops.name_scope(None, 'Input') as sc:
      metadata_handler = config.create_metadata_handler_fn(config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, self.length_label_batches, self.numbers_label_batches = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size)

    self.total_loss = None
    self.global_step = None

  def build(self):
    self._build_base_net()

    number_losses = []
    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(self.length_output, self.length_label_batches)
      length_loss = tf.log(tf.reduce_mean(length_loss), 'length_loss')
      self.total_loss = length_loss

      for i in range(self.config.max_number_length):
        number_loss = tf.nn.softmax_cross_entropy_with_logits(self.numbers_output[i], self.numbers_label_batches[:,i,:])
        number_loss = tf.log(tf.reduce_mean(number_loss), 'number%s_loss' % (i + 1))
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


class CNNEvalModel(CNNModelBase):
  pass


class CNNPredictModel(CNNModelBase):
  pass
