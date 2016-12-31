import os

import tensorflow as tf

from nsrec.data_preprocessor import main, real_main
from nsrec.inputs_test import DataReaderTest


class DataPreprocessorTest(tf.test.TestCase):

  def test_main(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestMatMetadata(25, metadata_dir_path)
    filenames, labels, bboxes = real_main(
      metadata_file_path,
      5,
      os.path.join(metadata_dir_path, 'metadata-expanded.pickle'),
      os.path.join(metadata_dir_path, '../../data/train'),
      rand_box_count=0)
    self.assertEqual(len(filenames), 25)
    filenames, labels, bboxes = real_main(
      metadata_file_path,
      5,
      os.path.join(metadata_dir_path, 'metadata-expanded.pickle'),
      os.path.join(metadata_dir_path, '../../data/train'),
      rand_box_count=5)
    self.assertEqual(len(filenames), 25 * 5)


if __name__ == '__main__':
  tf.test.main()