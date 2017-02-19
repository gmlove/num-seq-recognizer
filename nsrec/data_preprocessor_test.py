import os

from PIL import Image
from six.moves import cPickle as pickle

import tensorflow as tf
from nsrec import test_helper
from nsrec.data_preprocessor import parse_data


class DataPreprocessorTest(tf.test.TestCase):

  def test_main(self):
    metadata_file_path = test_helper.get_mat_test_metadata()
    filenames, labels, bboxes = parse_data(
      metadata_file_path, 5, test_helper.train_data_dir_path, rand_box_count=0)
    self.assertEqual(len(filenames), 25)
    filenames, labels, bboxes = parse_data(
      metadata_file_path, 5, test_helper.train_data_dir_path, rand_box_count=5)
    self.assertEqual(len(filenames), 25 * 5)


if __name__ == '__main__':
  tf.test.main()