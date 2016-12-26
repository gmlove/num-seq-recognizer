import h5py
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

from nsrec.models import BBox, Data


def metadata_generator(file_path):
  f = h5py.File(file_path)
  refs, ds = f['#refs#'], f['digitStruct']

  def bboxes(i):
    attr_names = 'label top left width height'.split()
    bboxes = []
    bboxes_raw = refs[ds['bbox'][i][0]]
    bboxes_count = bboxes_raw['label'].value.shape[0]
    for j in range(bboxes_count):
      real_value = lambda ref_or_real_value: refs[ref_or_real_value].value.reshape(-1)[0] \
        if isinstance(ref_or_real_value, h5py.h5r.Reference) else ref_or_real_value
      attr_value = lambda attr_name: real_value(bboxes_raw[attr_name].value[j][0])
      bboxes.append(BBox(*[attr_value(an) for an in attr_names]))
    return bboxes

  for i, name in enumerate(ds['name']):
    ords = refs[name[0]].value
    name_str = ''.join([chr(ord) for ord in ords.reshape(-1)])
    yield Data(name_str, bboxes(i))


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("mat_metadata_file_path", "",
                       "Mat format metadata file path.")
tf.flags.DEFINE_string("output_file_path", "",
                       "Output file path.")
tf.flags.DEFINE_integer("max_numbers_length", 5,
                       "Max numbers length.")


def main(args):
  assert FLAGS.mat_metadata_file_path, "Mat format metadata file path required"
  assert FLAGS.output_file_path, "Output file path required"

  metadata = metadata_generator(FLAGS.mat_metadata_file_path)
  filenames, labels = [], []
  for i, md in enumerate(metadata):
    if len(md.label) > FLAGS.max_numbers_length:
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