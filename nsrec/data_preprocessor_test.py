import os

from PIL import Image
from six.moves import cPickle as pickle

import tensorflow as tf
from nsrec import test_helper
from nsrec.data_preprocessor import parse_data


class DataPreprocessorTest(tf.test.TestCase):

  def test_main(self):
    metadata_file_path = test_helper.get_mat_test_metadata()
    filenames, labels, bboxes, sep_bboxes = parse_data(
      metadata_file_path, 5, test_helper.train_data_dir_path, rand_box_count=0)
    self.assertEqual(len(filenames), 25)
    self.assertEqual(len(sep_bboxes[0][0]), 4)
    self.assertEqual(len(labels[3]), len(sep_bboxes[3]))
    filenames, labels, bboxes, sep_bboxes = parse_data(
      metadata_file_path, 5, test_helper.train_data_dir_path, rand_box_count=5)
    self.assertEqual(len(filenames), 25 * 5)


if __name__ == '__main__':
  tf.test.main()