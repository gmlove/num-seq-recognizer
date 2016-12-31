import os
import random

import tensorflow as tf
from six.moves import cPickle as pickle
from PIL import Image

from nsrec.inputs import metadata_generator

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("mat_metadata_file_path", "",
                       "Mat format metadata file path.")
tf.flags.DEFINE_string("data_dir_path", "",
                       "Mat format metadata file path.")
tf.flags.DEFINE_string("output_file_path", "",
                       "Output file path.")
tf.flags.DEFINE_integer("max_number_length", 5,
                       "Max numbers length.")
tf.flags.DEFINE_bool("rand_bbox_count", 5,
                        "If generate rand bbox.")


def main(args, **kwargs):
  assert FLAGS.mat_metadata_file_path, "Mat format metadata file path required"
  assert FLAGS.output_file_path, "Output file path required"
  assert FLAGS.data_dir_path, "Data dir path required"

  filenames, labels, bboxes = real_main(FLAGS.mat_metadata_file_path, FLAGS.max_number_length,
        FLAGS.output_file_path, FLAGS.data_dir_path, FLAGS.rand_bbox_count)

  pickle.dump({'filenames': filenames, 'labels': labels, 'bboxes': bboxes},
              open(FLAGS.output_file_path, 'wb'))


def real_main(mat_metadata_file_path, max_number_length, output_file_path, data_dir_path,
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
  if bbox[0] + bbox[2] > im.size[0] or bbox[1] + bbox[3] > im.size[1]:
    tf.logging.info('fix record %s, bad bbox. bbox=%s, size=%s', md.filename, bbox, im.size)
    bbox[2] = im.size[0] - bbox[0] if bbox[0] + bbox[2] > im.size[0] else bbox[2]
    bbox[3] = im.size[1] - bbox[1] if bbox[1] + bbox[3] > im.size[1] else bbox[3]
    tf.logging.info('fix record %s to bbox=%s, size=%s', md.filename, bbox, im.size)
  return bbox, im.size


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

'''
how to test:
python nsrec/data_preprocessor.py --mat_metadata_file_path= ./nsrec/test_data/digitStruct.mat \
  --data_dir_path=./data/train --output_file_path=./nsrec/test_data/metadata-expanded.pickle
'''

if __name__ == '__main__':
  tf.app.run()