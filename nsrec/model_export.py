import tensorflow as tf
from nsrec.model_export_base import export

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("output_file_name", "graph.pb",
                       "Output file name.")


def main(_):
  export(FLAGS)

if __name__ == "__main__":
  tf.app.run()
