import os
import tensorflow as tf
import numpy as np
from nsrec import inputs
from nsrec.data_preprocessor_test import DataReaderTest


class InputTest(tf.test.TestCase):

  def test_batches_from_mat(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestMatMetadata(25, metadata_dir_path)

    self._test_batches(metadata_file_path, inputs.mat_metadata_handler)

  def test_batches_from_pickle(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestPickleMetadata(25, metadata_dir_path)

    self._test_batches(metadata_file_path, inputs.pickle_metadata_handler)

  def _test_batches(self, metadata_file_path, metadata_handler_fn):
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/train')

    max_number_length, batch_size, size = 5, 2, (28, 28)
    with self.test_session() as sess:
      metadata_handler = metadata_handler_fn(metadata_file_path, max_number_length, data_dir_path)
      data_batches, length_label_batches, numbers_label_batches = \
        inputs.batches(metadata_handler, max_number_length, batch_size, size)

      self.assertEqual(data_batches.get_shape(), (2, 28, 28, 3))
      self.assertEqual(length_label_batches.get_shape(), (2, max_number_length))
      self.assertEqual(numbers_label_batches.get_shape(), (2, max_number_length, 10))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      batches = []
      for i in range(5):
        batches.append(sess.run([data_batches, length_label_batches, numbers_label_batches]))

      db, llb, nlb = batches[0]
      one_hot = lambda num, max_num: np.eye(max_num)[num - 1]
      self.assertAllEqual(llb, one_hot(np.array([2, 2]), max_number_length))
      self.assertNDArrayNear(nlb[0], np.concatenate([
        one_hot(np.array([1, 9]) + 1, 10), np.array([[0.1] * 10] * 3)
      ]), 1e-5)

      coord.request_stop()
      coord.join(threads)
      sess.close()
