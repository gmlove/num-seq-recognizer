from tensorflow.contrib import slim
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_model
import inputs
from nets import rnn


class RNNTrainModel(cnn_model.CNNNSRTrainModel):

  def __init__(self, config):
    super(RNNTrainModel, self).__init__(config)
    self.embedding_size = 256
    self.max_number_length = self.config.max_number_length

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      metadata_handler = config.create_metadata_handler_fn(
        config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, _, numbers_label_batches = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training, channels=config.channels)
      numbers_label_batches = tf.transpose(numbers_label_batches, perm=[1, 0, 2])
      self.numbers_label_batches = tf.reshape(numbers_label_batches, [-1, 11])

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection = self.cnn_net.end_points_collection_name(variable_scope)
      net, _ = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection)
      net = slim.fully_connected(net, self.embedding_size, activation_fn=None, scope='fc0')
      net = rnn.rnn_layers(net, tf.ones((self.max_number_length, self.batch_size), dtype=tf.int32), self.embedding_size)
      net = tf.reshape(net, [-1, self.embedding_size])
      self.model_output = slim.fully_connected(net, 11, activation_fn=None, scope='fc4')

  def _setup_loss(self):
    losses = tf.nn.softmax_cross_entropy_with_logits(self.model_output, self.numbers_label_batches)

    batch_loss = tf.reduce_mean(losses)
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()

    # Add summaries.
    tf.summary.scalar("batch_loss", batch_loss)
    tf.summary.scalar("total_loss", total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    self.total_loss = total_loss

  def _setup_accuracy(self, op_name='accuracy/train'):
    cnn_model.softmax_accuracy(self.model_output, self.numbers_label_batches, op_name)


class RNNEvalModel(RNNTrainModel):
  pass