import tensorflow as tf
from nsrec import ArgumentsObj
from nsrec.model_export_base import export

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("output_file_name", "graph-yolo.pb",
                       "Output file name.")


def main(_):
  args = ArgumentsObj().define_arg('model_type', 'yolo')
  export(args)

if __name__ == "__main__":
  tf.app.run()
