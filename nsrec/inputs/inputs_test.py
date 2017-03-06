import numpy as np

import tensorflow as tf
from nsrec import inputs, test_helper
from nsrec.utils.np_ops import one_hot


class InputTest(tf.test.TestCase):

  def __init__(self, *args):
    super(InputTest, self).__init__(*args)

  def test_batches(self):
    numbers_labels = lambda numbers: np.concatenate(
      [one_hot(np.array(numbers) + 1, 11), np.array([one_hot(11, 11) for _ in range(5 - len(numbers))])])
    max_number_length, expected_length_labels, expected_numbers_labels, expected_numbers_labels_1 = \
        5, one_hot(np.array([2, 2]), 5), numbers_labels([1, 9]), numbers_labels([2, 3])

    data_file_path = test_helper.get_test_metadata()
    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      data_batches, length_label_batches, numbers_label_batches = \
        inputs.batches(data_file_path, max_number_length, batch_size, size, num_preprocess_threads=1, channels=3)

      self.assertEqual(data_batches.get_shape(), (2, 28, 28, 3))
      self.assertEqual(length_label_batches.get_shape(), (2, max_number_length))
      self.assertEqual(numbers_label_batches.get_shape(), (2, max_number_length, 11))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      batches = []
      for i in range(5):
        batches.append(sess.run([data_batches, length_label_batches, numbers_label_batches]))

      db, llb, nlb = batches[0]
      self.assertAllEqual(llb, expected_length_labels)
      self.assertNDArrayNear(nlb[0], expected_numbers_labels, 1e-5)
      self.assertNDArrayNear(nlb[1], expected_numbers_labels_1, 1e-5)

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def test_bbox_batches(self):
    # bbox of 1.png(19): 246, 77, 173, 223, size of 1.png: 741 x 350
    self._test_bbox_batches(None, [246 * 100 / 741, 77 * 100 / 350, 173 * 100 / 741, 223 * 100 / 350])

  def test_bbox_batches_for_sep_bbox_0(self):
    # separated bboxes of 1.png(19): [[246, 77, 81, 219], [323, 81, 96, 219]], size of 1.png: 741 x 350
    self._test_bbox_batches('sep_bbox_0', [246 * 100 / 741, 77 * 100 / 350, 81 * 100 / 741, 219 * 100 / 350])

  def test_bbox_batches_for_number_0(self):
    test_helper.get_test_metadata()
    # separated bboxes of 25.png(601): [[60, 11, 24, 50], [87, 9, 24, 50], [113, 7, 21, 50]], size of 1.png: 190 x 75
    self._test_bbox_batches('number_0', [87 * 100 / 190, 9 * 100 / 75, 24 * 100 / 190, 50 * 100 / 75],
                            test_helper.test_data_file_number_0)

  def _test_bbox_batches(self, target_bbox, first_expected_bbox, data_file_path=None):
    data_file_path = test_helper.get_test_metadata() if data_file_path is None else data_file_path
    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      data_batches, bbox_batches = \
        inputs.bbox_batches(data_file_path, batch_size, size, 5,
                            num_preprocess_threads=1, channels=3, target_bbox=target_bbox)

      self.assertEqual(data_batches.get_shape(), (2, 28, 28, 3))
      self.assertEqual(bbox_batches.get_shape(), (2, 4))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      _, bb = sess.run([data_batches, bbox_batches])
      self.assertAllClose(bb[0], first_expected_bbox)

      coord.request_stop()
      coord.join(threads)
      sess.close()
