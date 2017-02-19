import os

import h5py
import numpy as np
from six.moves import cPickle as pickle

import tensorflow as tf
from nsrec import inputs
from nsrec.inputs import metadata_generator
from nsrec.inputs.models import Data, BBox
from nsrec.utils.np_ops import one_hot


def relative_file(path):
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


class InputTest(tf.test.TestCase):
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
    metadata_file_path = self._test_metadata_file_path()
    self._test_batches(metadata_file_path)

  def _test_metadata_file_path(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    metadata_file_path = DataReaderTest.createTestPickleMetadata(25, metadata_dir_path)
    return metadata_file_path

  def test_batches_with_label_length_longer_than_max_num_length(self):
    self._test_batches(self._test_metadata_file_path(), 1,
                       one_hot(np.array([1, 1]), 1), np.array([[0, 1] + [0] * 9]), np.array([[0, 0, 1] + [0] * 8]))

  def _test_batches(self, metadata_file_path, max_number_length=5,
                    expected_length_labels=None, expected_numbers_labels=None,
                    expected_numbers_labels_1=None, data_dir_path=None):
    data_dir_path = data_dir_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/train')

    batch_size, size = 2, (28, 28)
    with self.test_session() as sess:
      metadata_handler = inputs.create_pickle_metadata_handler(metadata_file_path, max_number_length, data_dir_path)
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
      expected_length_labels = expected_length_labels if expected_length_labels is not None else one_hot(
        np.array([2, 2]), max_number_length)
      self.assertAllEqual(llb, expected_length_labels)
      self.assertNumbersLabelEqual(nlb, expected_numbers_labels, expected_numbers_labels_1)

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def test_bbox_batches(self):
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/train')
    batch_size, size = 2, (28, 28)
    max_number_length = 5
    with self.test_session() as sess:
      metadata_handler = inputs.create_pickle_metadata_handler(self._test_metadata_file_path(), max_number_length,
                                                               data_dir_path)
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

  def assertNumbersLabelEqual(self, nlb, expected_numbers_labels, expected_numbers_labels_1):
    def expected_numbers_label(_expected_numbers_labels, numbers):
      return _expected_numbers_labels if _expected_numbers_labels is not None else np.concatenate(
        [one_hot(np.array(numbers) + 1, 11), np.array([one_hot(11, 11) for i in range(5 - len(numbers))])])

    expected_numbers_labels = expected_numbers_label(expected_numbers_labels, [1, 9])
    self.assertNDArrayNear(nlb[0], expected_numbers_labels, 1e-5)
    expected_numbers_labels_1 = expected_numbers_label(expected_numbers_labels_1, [2, 3])
    self.assertNDArrayNear(nlb[1], expected_numbers_labels_1, 1e-5)

  def xx_test_non_zero_size_image_when_run_in_multi_thread(self):
    # TODO: fix this issue
    max_number_length, batch_size, size = 5, 32, (64, 64)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_file_path = os.path.join(current_dir, '../test_data/metadata.pickle')
    data_dir_path = os.path.join(current_dir, '../../data/train')
    with self.test_session() as sess:
      data_gen_fn = inputs.create_pickle_metadata_handler(metadata_file_path, max_number_length, data_dir_path)
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


class DataReaderTest(tf.test.TestCase):
  @classmethod
  def createTestPickleMetadata(cls, test_data_count, test_dir_path=None):
    mat_metadata_file = DataReaderTest.createTestMatMetadata(test_data_count, test_dir_path)

    filenames, labels, bboxes = [], [], []
    for i, data in enumerate(metadata_generator(mat_metadata_file)):
      filenames.append(data.filename)
      labels.append(data.label)
      bboxes.append(data.bbox())

    metadata_file = os.path.join(test_dir_path, 'metadata.pickle')
    with open(metadata_file, 'wb') as f:
      pickle.dump({'filenames': filenames, 'labels': labels, 'bboxes': bboxes}, f)

    return metadata_file

  @classmethod
  def createTestMatMetadata(cls, test_data_count, test_dir_path=None):
    test_dir_path = test_dir_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    metadata_file = os.path.join(test_dir_path, 'digitStruct.mat')
    test_f = h5py.File(metadata_file, 'w')

    f = h5py.File(os.path.join(test_dir_path, '../../data/train/digitStruct.mat'))
    refs, ds = f['#refs#'], f['digitStruct']

    t_ds = test_f.create_group('digitStruct')
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    t_refs = test_f.create_group('#refs#')

    data_idx = 0

    def create_t_real_data(ref):
      nonlocal data_idx
      real = refs[ref]
      if isinstance(real, h5py.Group):
        created_group = t_refs.create_group('data_%s' % data_idx)
        data_idx += 1
        attrs = 'label top left width height'.split()
        for attr in attrs:
          reshaped = real[attr].value.reshape(-1)
          data_count = reshaped.shape[0]
          if isinstance(reshaped[0], h5py.Reference):
            t_real_attr = created_group.create_dataset(attr, shape=(data_count, 1), dtype=ref_dtype)
            for i in range(data_count):
              t_real_attr[i, 0] = create_t_real_data(reshaped[i])
          else:
            created_group.create_dataset(attr, data=real[attr].value)
            data_idx += 1
        return created_group.ref
      else:
        t_real = t_refs.create_dataset('data_%s' % data_idx, data=real.value)
        data_idx += 1
        return t_real.ref

    def create_t_element(t_group, name, ref_group, data_count):
      reshaped = ref_group[name].value.reshape(-1)
      data_count = reshaped.shape[0] if data_count is None else data_count
      created_dataset = t_group.create_dataset(name, (data_count, 1), dtype=ref_dtype)
      for i in range(data_count):
        created_dataset[i, 0] = create_t_real_data(reshaped[i])

    create_t_element(t_ds, 'name', ds, test_data_count)
    create_t_element(t_ds, 'bbox', ds, test_data_count)
    test_f.close()
    return metadata_file

  def test_create_metadata_file_for_testing(self):
    DataReaderTest.createTestMatMetadata(25)

    data_pack = []
    for i, data in enumerate(metadata_generator(relative_file('../test_data/digitStruct.mat'))):
      data_pack.append(data)
    self.assertEqual(data_pack[0].label, '19')
    self.assertEqual(data_pack[20].label, '2')
    self.assertEqual(data_pack[24].label, '601')
    self.assertEqual(data_pack[24].filename, '25.png')

  def test_metadata_generator(self):
    gen = metadata_generator(relative_file('../../data/train/digitStruct.mat'))
    sampled = [gen.__next__() for i in range(30)]

    self.assertIsInstance(sampled[0], Data)
    self.assertEqual(sampled[0], Data('1.png', [BBox('1', 77, 246, 81, 219), BBox('9', 81, 323, 96, 219)]))
    self.assertEqual(sampled[20], Data('21.png', [BBox(label=2, top=6.0, left=72.0, width=52.0, height=85.0)]))
