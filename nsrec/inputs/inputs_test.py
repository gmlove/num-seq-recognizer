import os

import numpy as np

import tensorflow as tf
from nsrec import inputs, test_helper
from nsrec.utils.np_ops import one_hot


def relative_file(path):
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


class InputTest(tf.test.TestCase):

  def __init__(self, *args):
    super(InputTest, self).__init__(*args)

  def test_mnist_batches(self):
    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      data_batches, label_batches = inputs.mnist_batches(batch_size, size, data_count=100)

      self.assertEqual(data_batches.get_shape().as_list(), [2, 28, 28, 1])
      self.assertEqual(label_batches.get_shape().as_list(), [2, 10])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      batches = []
      for i in range(5):
        batches.append(sess.run([data_batches, label_batches]))

      _, lb = batches[0]
      self.assertAllEqual(np.argmax(lb, -1), [7, 3])

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def test_batches_from_pickle(self):
    numbers_labels = lambda numbers: np.concatenate(
      [one_hot(np.array(numbers) + 1, 11), np.array([one_hot(11, 11) for _ in range(5 - len(numbers))])])
    self._test_batches(5, one_hot(np.array([2, 2]), 5), numbers_labels([1, 9]), numbers_labels([2, 3]))

  def test_batches_with_label_length_longer_than_max_num_length(self):
    self._test_batches(1, one_hot(np.array([1, 1]), 1), np.array([[0, 1] + [0] * 9]), np.array([[0, 0, 1] + [0] * 8]))

  def _test_batches(self, max_number_length, expected_length_labels,
                    expected_numbers_labels, expected_numbers_labels_1):
    metadata_file_path = test_helper.get_test_metadata()
    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      metadata_handler = inputs.create_pickle_metadata_handler(
        metadata_file_path, max_number_length, test_helper.train_data_dir_path)
      data_batches, length_label_batches, numbers_label_batches = \
          inputs.batches(metadata_handler, max_number_length, batch_size, size, num_preprocess_threads=1, channels=3)

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

  def test_batches_from_tfrecords(self):
    numbers_labels = lambda numbers: np.concatenate(
      [one_hot(np.array(numbers) + 1, 11), np.array([one_hot(11, 11) for _ in range(5 - len(numbers))])])
    max_number_length, expected_length_labels, expected_numbers_labels, expected_numbers_labels_1 = \
        5, one_hot(np.array([2, 2]), 5), numbers_labels([1, 9]), numbers_labels([2, 3])

    metadata_file_path = test_helper.get_test_metadata()
    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      metadata_handler = inputs.create_pickle_metadata_handler(
        metadata_file_path, max_number_length, test_helper.train_data_dir_path)
      data_batches, length_label_batches, numbers_label_batches = \
        inputs.batches(metadata_handler, max_number_length, batch_size, size, num_preprocess_threads=1, channels=3)

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
    batch_size, size = 2, (28, 28)
    max_number_length = 5
    with self.test_session() as sess:
      metadata_file_path = test_helper.get_test_metadata()
      metadata_handler = inputs.create_pickle_metadata_handler(metadata_file_path, max_number_length,
                                                               test_helper.train_data_dir_path)
      data_batches, bbox_batches = \
        inputs.bbox_batches(metadata_handler, batch_size, size, num_preprocess_threads=1, channels=3)

      self.assertEqual(data_batches.get_shape(), (2, 28, 28, 3))
      self.assertEqual(bbox_batches.get_shape(), (2, 4))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # bbox of 1.png: 246, 77, 173, 223, size of 1.png: 741 x 350
      _, bb = sess.run([data_batches, bbox_batches])
      self.assertAllClose(bb[0], [246 / 741, 77 / 350, 173 / 741, 223 / 350])

      coord.request_stop()
      coord.join(threads)
      sess.close()


  def xx_test_non_zero_size_image_when_run_in_multi_thread(self):
    # TODO: fix this issue
    max_number_length, batch_size, size = 5, 32, (64, 64)
    with self.test_session() as sess:
      data_gen_fn = inputs.create_pickle_metadata_handler(
        self.metadata_file_path, max_number_length, test_helper.train_data_dir_path)
      old_fn = inputs.resize_image

      def handle_image(dequeued_img, dequeued_bbox, *args):
        # dequeued_img1 = dequeued_img[
        #   dequeued_bbox[1]:dequeued_bbox[1]+dequeued_bbox[3], dequeued_bbox[0]:dequeued_bbox[0]+dequeued_bbox[2], :
        # ]
        dequeued_img2 = tf.image.resize_images(dequeued_img, [size[0], size[1]])
        # return tf.concat_v2([tf.shape(dequeued_img2), tf.shape(dequeued_img), dequeued_bbox], 0)
        return tf.concat_v2(
          [tf.cast(tf.expand_dims(tf.reduce_sum(dequeued_img2), 0), dtype=tf.int32), tf.shape(dequeued_img2),
           tf.shape(dequeued_img), dequeued_bbox], 0)

      inputs.resize_image = handle_image
      data_batches, _, _ = inputs.batches(data_gen_fn, max_number_length, batch_size, size, num_preprocess_threads=3)
      inputs.resize_image = old_fn
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for i in range(1000):
        output = sess.run(data_batches)
        print('image.shape=%s' % output)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


