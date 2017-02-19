import os

import tensorflow as tf
from models.cnn_model import create_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("number_of_steps", 5000, "Number of training steps.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size.")

default_data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/train.raw.tfrecords')
tf.flags.DEFINE_string("data_file_path", default_data_file_path, "Data file path.")

tf.flags.DEFINE_integer("save_summaries_secs", 5, "Save summaries per secs.")
tf.flags.DEFINE_integer("save_interval_secs", 180, "Save model per secs.")

tf.flags.DEFINE_string("optimizer", "SGD", "Optimizer: SGD")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate")
tf.flags.DEFINE_integer("max_checkpoints_to_keep", 5, "Max checkpoints to keep")
current_dir = os.path.dirname(os.path.abspath(__file__))
tf.flags.DEFINE_string("train_dir", os.path.join(current_dir, '../output/train'), "Train output directory.")
tf.flags.DEFINE_integer("num_preprocess_threads", 5, "Number of pre-processor threads")

def learning_rate_fn(batch_size):
  num_epochs_per_decay = 8.0
  learning_rate = tf.constant(FLAGS.learning_rate)
  num_batches_per_epoch = (10000 / batch_size)
  decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

  def learning_rate_decay_fn(learning_rate, global_step):
    return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=decay_steps,
      decay_rate=0.5,
      staircase=True)

  return learning_rate, learning_rate_decay_fn


def main(unused_argv):
  train_dir = FLAGS.train_dir
  if not os.path.exists(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    os.makedirs(train_dir)

  g = tf.Graph()
  with g.as_default():
    model = create_model(FLAGS)
    model.build()

    learning_rate, learning_rate_decay_fn = learning_rate_fn(model.config.batch_size)

    train_op = tf.contrib.layers.optimize_loss(
      loss=model.total_loss,
      global_step=model.global_step,
      learning_rate=learning_rate,
      learning_rate_decay_fn=learning_rate_decay_fn,
      optimizer=FLAGS.optimizer)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)

  tf.contrib.slim.learning.train(
    train_op,
    train_dir,
    log_every_n_steps=FLAGS.log_every_n_steps,
    graph=g,
    global_step=model.global_step,
    number_of_steps=FLAGS.number_of_steps,
    save_interval_secs=FLAGS.save_interval_secs,
    save_summaries_secs=FLAGS.save_summaries_secs,
    saver=saver)

if __name__ == '__main__':
  tf.app.run()
