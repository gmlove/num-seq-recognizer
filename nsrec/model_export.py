import os

import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.cnn_model import create_model
from six.moves import cPickle as pickle

FLAGS = tf.flags.FLAGS

current_dir = os.path.dirname(os.path.abspath(__file__))
default_checkpoint_dir = os.path.join(current_dir, '../output/train')

tf.flags.DEFINE_string("output_file_path", "./graph.pb",
                       "Output file path.")

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    model = create_model(FLAGS, 'inference')
    model.build()
    saver = tf.train.Saver()

  g.finalize()

  model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping inference. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
    return


  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", FLAGS.checkpoint_dir)
    saver.restore(sess, model_path)
    model_vars = model.vars(sess)

  # Build graph to export
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    model = create_model(FLAGS, 'to_export')
    model.init(model_vars)
    model.build()

  with tf.Session(graph=g) as sess:
    log_dir = './output/export'
    if not tf.gfile.IsDirectory(log_dir):
      tf.logging.info("Creating log directory: %s", log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.train.write_graph(sess.graph_def, log_dir, 'graph.pb', as_text=False)

if __name__ == "__main__":
  tf.app.run()
