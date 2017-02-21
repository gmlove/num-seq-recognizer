import tensorflow as tf
from nsrec import inputs
from nsrec.nets import basic_net
from nsrec.utils.ops import global_step_variable, softmax_accuracy, softmax_cross_entrophy_loss
from tensorflow.python.framework import ops


class CNNMnistTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.model_output = None
    self.label_batches = None
    self.is_training = True

  def _setup_input(self):
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches = \
        inputs.mnist_batches(self.config.batch_size, self.config.size, is_training=self.is_training)

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, self.is_training)

  def _setup_loss(self):
    self.total_loss = softmax_cross_entrophy_loss(self.model_output, self.label_batches)

  def _setup_accuracy(self):
    self.train_accuracy = softmax_accuracy(self.model_output, self.label_batches, 'accuracy/train')

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_loss()
    self._setup_accuracy()
    self._setup_global_step()
