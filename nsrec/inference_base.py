import os

from six.moves import cPickle as pickle

import tensorflow as tf
from nsrec.models import create_model
from nsrec import inputs

_FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
current_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir_path = os.path.join(current_dir, '../data/train')
default_metadata_file_path = os.path.join(default_data_dir_path, 'metadata.pickle')
tf.flags.DEFINE_string("metadata_file_path", default_metadata_file_path, "Metadata file path.")
tf.flags.DEFINE_string("data_dir_path", default_data_dir_path, "Data file path.")
tf.flags.DEFINE_float("bbox_expand", 0.0, "Expand bbox to add extra pixels, "
                                          "set to 0.1 will reduce 8% of accuracy for training dataset"
                                          "and 4% for extra dataset.")


def inference(label_fn, bboxes=False, flags=None):
  """
  Used to infer against a nsr bbox model or a nsr model.
  Args:
    label_fn: Accept parameters (label, metadata_bbox), and return the real label
    bboxes: Use metadata bbox if False. Do not crop image if None. Will crop image if not False and None.
    flags:

  Returns:

  """
  flags = flags or _FLAGS
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

    files = [s.strip() for s in flags.input_files.split(',')]
    metadata = pickle.loads(open(flags.metadata_file_path, 'rb').read())
    real_labels = []

    file_paths = [os.path.join(flags.data_dir_path, f) for f in files]
    data = []
    for i, f in enumerate(files):
      metadata_idx = metadata['filenames'].index(f)
      label, metadata_bbox = metadata['labels'][metadata_idx], metadata['bboxes'][metadata_idx]
      real_labels.append(label_fn(label, metadata_bbox))
      bbox = (bboxes[i] if bboxes is not None else None) if bboxes is not False else metadata_bbox
      data.append(inputs.read_img(file_paths[i], bbox, flags.bbox_expand))

    labels = model.infer(sess, data)
    for i in range(len(files)):
      tf.logging.info('inferred image %s[%s]: %s', files[i], real_labels[i], labels[i])
    correct_inferences = filter(lambda i: real_labels[i] == labels[i][0], range(len(files)))
    correct_count = len(list(correct_inferences))
    tf.logging.info('correct count: %s, rate: %.4f', correct_count, correct_count / len(files))
    return labels
