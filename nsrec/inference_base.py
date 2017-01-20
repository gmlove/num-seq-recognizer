import os

from six.moves import cPickle as pickle

import tensorflow as tf
from models.cnn_model import create_model
from nsrec import inputs, ArgumentsObj

_FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

def combined_inference():
  args = ArgumentsObj('bbox').defineArg('cnn_model_type', 'bbox')
  bboxes = inference(lambda labels, bboxes: bboxes, None, args)
  if not bboxes:
    raise Exception("Bbox not calculated.")
  inference(lambda labels, bboxes: labels, bboxes)

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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(current_dir, '../data/train')
    metadata_file_path = os.path.join(data_dir_path, 'metadata.pickle')

    files = [s.strip() for s in FLAGS.input_files.split(',')]
    metadata = pickle.loads(open(metadata_file_path, 'rb').read())
    should_be_labels = []

    file_paths = [os.path.join(data_dir_path, f) for f in files]
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
    return labels
