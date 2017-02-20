import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops

from nsrec.models.model_helper import softmax_accuracy
from nsrec.models.nsr_model import CNNNSRTrainModel
from nsrec import inputs
from nsrec.nets import rnn


class RNNTrainModel(CNNNSRTrainModel):

  def __init__(self, config):
    super(RNNTrainModel, self).__init__(config)
    self.embedding_size = 256
    self.max_number_length = self.config.max_number_length

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


class RNNEvalModel(RNNTrainModel):
  pass