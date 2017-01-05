import os

import tensorflow as tf
from PIL import Image
from six.moves import cPickle as pickle

from nsrec.data_preprocessor import main, real_main
from nsrec.inputs_test import DataReaderTest


class DataPreprocessorTest(tf.test.TestCase):

  def test_main(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestMatMetadata(25, metadata_dir_path)
    filenames, labels, bboxes = real_main(
      metadata_file_path,
      5,
      os.path.join(metadata_dir_path, '../../data/train'),
      rand_box_count=0)
    self.assertEqual(len(filenames), 25)
    filenames, labels, bboxes = real_main(
      metadata_file_path,
      5,
      os.path.join(metadata_dir_path, '../../data/train'),
      rand_box_count=5)
    self.assertEqual(len(filenames), 25 * 5)

  def test_check_bbox(self):
    metadata_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/test/metadata.pickle')
    metadata = pickle.load(open(metadata_file_path, 'rb'))

    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/test')
    files = os.listdir(data_dir_path)
    files = [f for f in files if f.find('.png') != -1]
    files_size = {}
    for f in files:
      im = Image.open(os.path.join(data_dir_path, os.path.join(data_dir_path, f)))
      files_size[f] = im.size

    for i, f in enumerate(metadata['filenames']):
      bbox = metadata['bboxes'][i]
      size = files_size[f]

      if any([j < 1 for j in bbox]) or bbox[0] > size[0] or bbox[1] > size[1]\
          or bbox[0] + bbox[2] > size[0] or bbox[1] + bbox[3] > size[1]\
          or bbox[2] <= 0 or bbox[3] <= 0:
        print('found bad record %s, bad bbox. bbox=%s, size=%s' % (f, bbox, size))



if __name__ == '__main__':
  tf.test.main()