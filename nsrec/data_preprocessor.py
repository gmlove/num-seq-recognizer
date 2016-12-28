import tensorflow as tf
from six.moves import cPickle as pickle

from nsrec.inputs import metadata_generator

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("mat_metadata_file_path", "",
                       "Mat format metadata file path.")
tf.flags.DEFINE_string("output_file_path", "",
                       "Output file path.")
tf.flags.DEFINE_integer("max_number_length", 5,
                       "Max numbers length.")


def main(args):
  assert FLAGS.mat_metadata_file_path, "Mat format metadata file path required"
  assert FLAGS.output_file_path, "Output file path required"

  metadata = metadata_generator(FLAGS.mat_metadata_file_path)
  filenames, labels = [], []
  for i, md in enumerate(metadata):
    if len(md.label) > FLAGS.max_number_length:
      tf.logging.info('ignore record, label too long: label=%s, filename=%s', md.label, md.filename)
      continue
    filenames.append(md.filename)
    labels.append(md.label)
    if i % 1000 == 0 and i > 0:
      tf.logging.info('readed count: %s', i)

  pickle.dump({'filenames': filenames, 'labels': labels},
              open(FLAGS.output_file_path, 'wb'))

if __name__ == '__main__':
  tf.app.run()