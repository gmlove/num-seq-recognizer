import os

from six.moves import cPickle as pickle

import tensorflow as tf
from models.cnn_model import create_model
from nsrec import inputs, ArgumentsObj

_FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
current_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir_path = os.path.join(current_dir, '../data/train')
default_metadata_file_path = os.path.join(default_data_dir_path, 'metadata.pickle')
tf.flags.DEFINE_string("metadata_file_path", default_metadata_file_path, "Metadata file path.")
tf.flags.DEFINE_string("data_dir_path", default_data_dir_path, "Data file path.")


def inference(label_fn, bboxes=False, FLAGS=None):
  FLAGS = FLAGS or _FLAGS
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

    files = [s.strip() for s in FLAGS.input_files.split(',')]
    metadata = pickle.loads(open(FLAGS.metadata_file_path, 'rb').read())
    should_be_labels = []

    file_paths = [os.path.join(FLAGS.data_dir_path, f) for f in files]
    data = []
    for i, f in enumerate(files):
      metadata_idx = metadata['filenames'].index(f)
      label, metadata_bbox = metadata['labels'][metadata_idx], metadata['bboxes'][metadata_idx]
      should_be_labels.append(label_fn(label, metadata_bbox))
      bbox = (bboxes[i] if bboxes is not None else None) if bboxes is not False else metadata_bbox
      data.append(inputs.read_img(file_paths[i], bbox))

    labels = model.infer(sess, data)
    for i in range(len(files)):
      tf.logging.info('infered image %s[%s]: %s', files[i], should_be_labels[i], labels[i])
    correct_inferences = filter(lambda i: should_be_labels[i] == labels[i][0], range(len(files)))
    correct_count = len(list(correct_inferences))
    tf.logging.info('correct count: %s, rate: %.4f', correct_count, correct_count / len(files))
    return labels
