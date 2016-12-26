import os
import tensorflow as tf

from nsrec import inputs
from nsrec.nets import alexnet


class CNNModelConfig(object):

  def __init__(self, **kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.data_dir_path = os.path.join(current_dir, '../data/train')
    self.metadata_file_path = os.path.join(self.data_dir_path, 'digitStruct.mat')
    self.max_number_length = 5
    self.batch_size = 64
    self.size = [alexnet.image_height, alexnet.image_width]
    self.train_dir = os.path.join(current_dir, '../output')

    for attr in ['data_dir_path', 'metadata_file_path', 'max_number_length', 'batch_size', 'size']:
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))


class CNNModelBase:

  def __init__(self, config):
    self.config = config
    self.variables = None
    self.data_batches = None

  def _build_base_net(self):
    with alexnet.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection = alexnet.end_points_collection_name(variable_scope)
      net, _ = alexnet.cnn_layers(self.data_batches, variable_scope, end_points_collection)
      length_output, _ = alexnet.fc_layers(net, variable_scope, end_points_collection,
                                           num_classes=self.config.max_number_length,
                                           name_prefix='length')
      numbers_output = []
      for i in range(self.config.max_number_length):
        number_output, _ = alexnet.fc_layers(net, variable_scope, end_points_collection,
                                             num_classes=10, name_prefix='number%s' % (i + 1))
        numbers_output.append(number_output)

      return length_output, numbers_output


class CNNTrainModel(CNNModelBase):

  def __init__(self, config):
    self.config = config
    self.data_batches, self.length_label_batches, self.numbers_label_batches = \
      inputs.batches(config.metadata_file_path, config.data_dir_path,
                    config.max_number_length, config.batch_size, config.size)
    self.total_loss = None
    self.global_step = None

  def build(self):
    length_output, numbers_output = self._build_base_net()
    length_loss = tf.nn.softmax_cross_entropy_with_logits(length_output, self.length_label_batches)
    tf.contrib.losses.add_loss(tf.log(tf.reduce_mean(length_loss), 'length_loss'))
    for i in range(self.config.max_number_length):
      number_loss = tf.nn.softmax_cross_entropy_with_logits(numbers_output[i], self.numbers_label_batches[:,i,:])
      tf.contrib.losses.add_loss(tf.log(tf.reduce_mean(number_loss), 'number%s_loss' % (i + 1)))
    self.total_loss = tf.contrib.losses.get_total_loss()

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
