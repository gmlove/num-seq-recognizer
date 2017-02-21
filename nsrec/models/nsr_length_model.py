from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import basic_net
from nsrec.utils.ops import softmax_accuracy, softmax_cross_entrophy_loss, global_step_variable


class CNNLengthTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net

    self.data_batches = None
    self.label_batches = None

    self.total_loss = None
    self.global_step = None
    self.train_accuracy = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches, _ = \
        inputs.batches(config.data_file_path, config.max_number_length, config.batch_size, config.size,
                       is_training=True, channels=self.config.channels,
                       num_preprocess_threads=self.config.num_preprocess_threads)

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.max_number_length, True)

  def _setup_accuracy(self):
    self.train_accuracy = softmax_accuracy(self.model_output, self.label_batches, 'accuracy/train')

  def _setup_loss(self):
    self.total_loss = softmax_cross_entrophy_loss(self.model_output, self.label_batches)

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_accuracy()
    self._setup_loss()
    self._setup_global_step()