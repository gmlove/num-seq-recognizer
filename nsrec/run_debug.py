import os.path

import tensorflow as tf
from models.cnn_model import create_model
from nsrec.debug import inspect_tensors

FLAGS = tf.flags.FLAGS

current_dir = os.path.dirname(os.path.abspath(__file__))
default_metadata_file_path = os.path.join(current_dir, '../data/test/metadata.pickle')
tf.flags.DEFINE_string("metadata_file_path", default_metadata_file_path,
                       "Meta data file path.")
default_data_dir_path = os.path.join(current_dir, '../data/test/')
tf.flags.DEFINE_string("data_dir_path", default_data_dir_path,
                       "Meta data directory path.")

tf.flags.DEFINE_string("cnn_model_type", "all", "Model type. all: approximate all numbers; length: only approximate length")
tf.flags.DEFINE_string("net_type", "lenet", "Which net to use: lenet or alexnet")
tf.flags.DEFINE_integer("max_number_length", 5, "Max number length.")
tf.flags.DEFINE_integer("batch_size", 2, "Batch size.")
tf.flags.DEFINE_integer("total_steps", 5, "Total steps to run.")

tf.logging.set_verbosity(getattr(tf.logging, 'DEBUG'))


def main(unused_args):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    # Build the model for evaluation.
    model = create_model(FLAGS, 'eval')
    model.build()

    with tf.Session() as sess:
      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      # Run evaluation on the latest checkpoint.
      try:
        for i in range(FLAGS.total_steps):
          inspect_tensors(sess)
      except Exception as e:  # pylint: disable=broad-except
        tf.logging.error("Evaluation failed.")
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=1)




if __name__ == "__main__":
  tf.app.run()
