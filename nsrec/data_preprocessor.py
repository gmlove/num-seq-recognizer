import os
import random

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from PIL import Image

from nsrec.inputs import metadata_generator

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("metadata_file_path", "",
                       "Metadata file path, use ',' to separate multiple files, suffix must be `pickle` or `mat`.")
tf.flags.DEFINE_string("data_dir_path", "",
                       "Data file path, use ',' to separate multiple paths, "
                       "should be in the same order as mat_metadata_file_path.")
tf.flags.DEFINE_string("final_data_dir_path", "",
                       "Mat format metadata file path.")
tf.flags.DEFINE_string("output_file_path", "",
                       "Output file path.")
tf.flags.DEFINE_bool("rand_bbox_count", 5, "How many rand bbox to generate.")


def main(args, **kwargs):
  assert FLAGS.metadata_file_path, "Metadata file path required"
  assert FLAGS.output_file_path, "Output file path required"
  assert FLAGS.data_dir_path, "Data dir path required"

  metadata_file_paths = FLAGS.metadata_file_path.split(',')
  data_dir_paths = FLAGS.data_dir_path.split(',')
  assert len(metadata_file_paths) == len(data_dir_paths)

  final_filenames, final_labels, final_bboxes = [], [], []

  for i in range(len(metadata_file_paths)):
    metadata_file_path, data_dir_path = metadata_file_paths[i], data_dir_paths[i]
    filenames, labels, bboxes = _read_meta_data(
      data_dir_path, metadata_file_path, FLAGS.max_number_length, FLAGS.rand_bbox_count)

    data_dir_path_last_section = data_dir_path.split('/')[-1]
    data_dir_path_last_section = data_dir_path_last_section or data_dir_path.split('/')[-2]
    final_filenames.extend([data_dir_path_last_section + '/' + fn for fn in filenames])
    final_bboxes.extend(bboxes)
    final_labels.extend(labels)

  output_file_path = FLAGS.output_file_path
  if output_file_path.endswith('.pickle'):
    write_pickle(final_bboxes, final_filenames, final_labels, output_file_path)
  elif output_file_path.endswith('.tfrecords'):
    write_tf_records(final_filenames, final_labels, FLAGS.max_number_length, final_bboxes,
                     data_dir_paths[0][:data_dir_paths[0].rfind('/')], output_file_path)
  else:
    raise Exception('output_file_path must end with .pickle or .tfrecords: %s' % output_file_path)


def _read_meta_data(data_dir_path, metadata_file_path, max_number_length, rand_bbox_count):
  if metadata_file_path.endswith('.mat'):
    return parse_data(metadata_file_path, max_number_length,
                      data_dir_path, rand_bbox_count)
  elif metadata_file_path.endswith('.pickle'):
    metadata = pickle.loads(open(metadata_file_path, 'rb').read())
    return metadata['filenames'], metadata['labels'], metadata['bboxes']


def write_pickle(final_bboxes, final_filenames, final_labels, output_file_path):
  print('Writing %s records to file %s' % (len(final_filenames), output_file_path))
  pickle.dump({'filenames': final_filenames, 'labels': final_labels, 'bboxes': final_bboxes},
              open(output_file_path, 'wb'))


def parse_data(mat_metadata_file_path, max_number_length, data_dir_path,
               rand_box_count=5):
  metadata = metadata_generator(mat_metadata_file_path)
  filenames, labels, bboxes = [], [], []
  for i, md in enumerate(metadata):
    if len(md.label) > max_number_length:
      tf.logging.info('ignore record, label too long: label=%s, filename=%s', md.label, md.filename)
      continue

    bbox, size = fix_bbox(md, data_dir_path)

    if any([j < 0 for j in bbox]) or bbox[0] >= size[0] or bbox[1] >= size[1]:
      tf.logging.info('ignore failed to fix record %s, bad bbox. bbox=%s', md.filename, bbox)
      continue

    if rand_box_count != 0:
      rand_bboxes = random_bbox(rand_box_count, bbox, size)
      for i in range(rand_box_count):
        filenames.append(md.filename)
        labels.append(md.label)
        bboxes.append(rand_bboxes[i])
    else:
      filenames.append(md.filename)
      labels.append(md.label)
      bboxes.append(bbox)

    if i % 1000 == 0 and i > 0:
      tf.logging.info('readed count: %s', i)

  rand_idxes = list(range(len(filenames)))
  random.shuffle(rand_idxes)
  rand_filenames, rand_labels, rand_bboxes = [], [], []
  for i in rand_idxes:
    rand_filenames.append(filenames[i])
    rand_labels.append(labels[i])
    rand_bboxes.append(bboxes[i])

  return rand_filenames, rand_labels, rand_bboxes


def fix_bbox(md, data_dir_path):
  bbox = list(md.bbox())
  if any([j < 0 for j in bbox]):
    tf.logging.info('fix record %s, bad bbox. bbox=%s', md.filename, bbox)
    bbox = [max(k, 0) for k in bbox]
    tf.logging.info('fix record %s to bbox=%s', md.filename, bbox)
  im = Image.open(os.path.join(data_dir_path, md.filename))
  im_size = im.size
  im.close()
  if bbox[0] + bbox[2] > im_size[0] or bbox[1] + bbox[3] > im_size[1]:
    tf.logging.info('fix record %s, bad bbox. bbox=%s, size=%s', md.filename, bbox, im_size)
    bbox[2] = im_size[0] - bbox[0] if bbox[0] + bbox[2] > im_size[0] else bbox[2]
    bbox[3] = im_size[1] - bbox[1] if bbox[1] + bbox[3] > im_size[1] else bbox[3]
    tf.logging.info('fix record %s to bbox=%s, size=%s', md.filename, bbox, im_size)
  return bbox, im_size


def random_bbox(count, bbox, size):
  def expand(x, x1, x2, percent):
    may_add = int((x2 - x1) * percent)
    right = min(x2 + may_add // 2, x)
    left = max(x1 - (may_add - (right - x2)), 0)
    return left, right

  expand_percentage = 0.3
  left, right = expand(size[0], bbox[0], bbox[0] + bbox[2], expand_percentage)
  top, bottom = expand(size[1], bbox[1], bbox[1] + bbox[3], expand_percentage)

  bboxes = []
  for i in range(count):
    rleft = random.randint(left, right - bbox[2])
    rtop = random.randint(top, bottom - bbox[3])
    bboxes.append([rleft, rtop, bbox[2], bbox[3]])
  return bboxes


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _normalize_label(label, max_number_length):
  label = [int(i) for i in label]
  label = label[:max_number_length]
  normalized = np.array([10] * max_number_length)
  normalized[0: len(label)] = label
  return normalized


def write_tf_records(filenames, labels, max_number_length, bboxes, data_dir_path, output_file_path):
  print('Writing %s records to file %s' % (len(filenames), output_file_path))
  writer = tf.python_io.TFRecordWriter(output_file_path)

  labels_lengths = [min(max_number_length, len(l)) for l in labels]
  normalized_labels = [_normalize_label(l, max_number_length) for l in labels]

  for index in range(len(filenames)):
    image_png = open(os.path.join(data_dir_path, filenames[index]), 'rb').read()
    example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int64_feature(normalized_labels[index]),
      'length': _int64_feature([labels_lengths[index]]),
      'bbox': _int64_feature(bboxes[index]),
      'image_png': _bytes_feature(image_png)}))
    writer.write(example.SerializeToString())
  writer.close()

'''
how to test:
python3 nsrec/data_preprocessor.py --metadata_file_path=./nsrec/test_data/digitStruct.mat,./nsrec/test_data/metadata.pickle \
  --data_dir_path=./data/train,./data/train --output_file_path=./nsrec/test_data/data.tfrecords
'''

if __name__ == '__main__':
  tf.app.run()
