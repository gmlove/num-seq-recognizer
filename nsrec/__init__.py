import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('log_level', 'info', 'Log level.')

tf.logging.set_verbosity(getattr(tf.logging, FLAGS.log_level.upper()))