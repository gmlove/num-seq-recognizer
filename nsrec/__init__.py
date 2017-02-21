import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('log_level', 'info', 'Log level.')

tf.logging.set_verbosity(getattr(tf.logging, FLAGS.log_level.upper()))

current_dir = os.path.dirname(os.path.abspath(__file__))

tf.flags.DEFINE_string("cnn_model_type", "all",
                       "Model type. all: approximate all numbers; length: only approximate length")
tf.flags.DEFINE_string("net_type", "lenet", "Which net to use: lenet or alexnet")
tf.flags.DEFINE_bool("rnn", False, "Whether to use rnn as the output layer")
tf.flags.DEFINE_integer("max_number_length", 5, "Max number length.")
tf.flags.DEFINE_bool("gray_scale", False, "If read image as gray scale image.")
default_checkpoint_dir = os.path.join(current_dir, '../output/train')
tf.flags.DEFINE_string("checkpoint_dir", default_checkpoint_dir,
                       "Directory containing model checkpoints.")


class ArgumentsObj(object):

  def __init__(self, prefix=''):
    self.prefix = prefix
    self.predefined_arguments = {}

  def define_arg(self, name, value):
    self.predefined_arguments[self.prefix + '_' + name] = value
    return self

  def __getattr__(self, item):
    key = self.prefix + '_' + item
    if key in self.predefined_arguments:
      return self.predefined_arguments[key]
    if hasattr(tf.flags.FLAGS, key):
      return getattr(tf.flags.FLAGS, key)
    return getattr(tf.flags.FLAGS, item)
