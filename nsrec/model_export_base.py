import os

import tensorflow as tf
from models.cnn_model import create_model


def export(FLAGS):
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
    model.build(model_vars)

  with tf.Session(graph=g) as sess:
    log_dir = './output/export'
    if not tf.gfile.IsDirectory(log_dir):
      tf.logging.info("Creating log directory: %s", log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.train.write_graph(sess.graph_def, log_dir, FLAGS.output_file_name, as_text=False)
