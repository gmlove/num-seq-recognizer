import tensorflow as tf
from models import create_model

tf.flags.DEFINE_boolean("finalize_graph", False,
                        "If finalize the exported graph. If true, initializer will not be available.")


def export(flags):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    model = create_model(flags, 'inference')
    model.build()
    saver = tf.train.Saver()

  g.finalize()

  model_path = tf.train.latest_checkpoint(flags.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping inference. No checkpoint found in: %s",
                    flags.checkpoint_dir)
    return

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", flags.checkpoint_dir)
    saver.restore(sess, model_path)
    model_vars = model.vars(sess)

  # Build graph to export
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    model = create_model(flags, 'to_export')
    model.build(model_vars)

  with tf.Session(graph=g) as sess:
    graph_def = sess.graph_def
    if flags.finalize_graph:
      sess.run(model.initializer)
      g.finalize()
      graph_def = g.as_graph_def()
      graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, model.output_names())

    log_dir = './output/export'
    if not tf.gfile.IsDirectory(log_dir):
      tf.logging.info("Creating log directory: %s", log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.train.write_graph(graph_def, log_dir, flags.output_file_name, as_text=False)
