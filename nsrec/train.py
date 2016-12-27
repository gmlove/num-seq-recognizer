import os

import tensorflow as tf

from nsrec.cnn_model import CNNModelConfig, CNNTrainModel

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("number_of_steps", 5000, "Number of training steps.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size.")

default_metadata_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/metadata.pickle')
tf.flags.DEFINE_string("metadata_file_path", default_metadata_file_path, "Meta data file path.")

tf.flags.DEFINE_integer("save_summaries_secs", 30, "Save summaries per secs.")

tf.flags.DEFINE_string("net_type", "lenet", "Which net to use: lenet or alexnet")


class TrainConfig():

  def __init__(self):

    # Optimizer for training the model.
    self.optimizer = "SGD"

    self.learning_rate = 0.5

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5

    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.train_dir = os.path.join(current_dir, '../output/train')



def main(unused_argv):
  model_config = CNNModelConfig(metadata_file_path=FLAGS.metadata_file_path,
                                batch_size=FLAGS.batch_size,
                                net_type=FLAGS.net_type)
  training_config = TrainConfig()

  if not os.path.exists(training_config.train_dir):
    tf.logging.info("Creating training directory: %s", training_config.train_dir)
    os.makedirs(training_config.train_dir)

  g = tf.Graph()
  with g.as_default():
    model = CNNTrainModel(model_config)
    model.build()

    train_op = tf.contrib.layers.optimize_loss(
      loss=model.total_loss,
      global_step=model.global_step,
      learning_rate=training_config.learning_rate,
      optimizer=training_config.optimizer)

    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  tf.contrib.slim.learning.train(
    train_op,
    training_config.train_dir,
    log_every_n_steps=FLAGS.log_every_n_steps,
    graph=g,
    global_step=model.global_step,
    number_of_steps=FLAGS.number_of_steps,
    save_summaries_secs=FLAGS.save_summaries_secs,
    saver=saver)

if __name__ == '__main__':
  tf.app.run()
