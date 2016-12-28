import os

import h5py
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

from nsrec import inputs
from nsrec.inputs import metadata_generator
from nsrec.models import Data, BBox
from nsrec.np_ops import one_hot


class InputTest(tf.test.TestCase):

  def test_batches_from_mat(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestMatMetadata(25, metadata_dir_path)

    self._test_batches(metadata_file_path, inputs.create_mat_metadata_handler)

  def test_batches_from_pickle(self):
    metadata_file_path = self._test_metadata_file_path()

    self._test_batches(metadata_file_path, inputs.create_pickle_metadata_handler)

  def _test_metadata_file_path(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestPickleMetadata(25, metadata_dir_path)
    return metadata_file_path

  def test_batches_with_label_length_longer_than_max_num_length(self):
    self._test_batches(self._test_metadata_file_path(), inputs.create_pickle_metadata_handler, 1,
                       one_hot(np.array([1, 1]), 1), np.array([[0, 1] + [0] * 8]), np.array([[0, 0, 1] + [0] * 7]))

  def _test_batches(self, metadata_file_path, metadata_handler_fn, max_number_length=5,
                    expected_length_labels=None, expected_numbers_labels=None, expected_numbers_labels_1=None):
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/train')

    batch_size, size = 2, (28, 28)
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
      expected_length_labels = expected_length_labels if expected_length_labels is not None else one_hot(np.array([2, 2]), max_number_length)
      self.assertAllEqual(llb, expected_length_labels)
      self.assertNumbersLabelEqual(nlb, expected_numbers_labels, expected_numbers_labels_1)

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def assertNumbersLabelEqual(self, nlb, expected_numbers_labels, expected_numbers_labels_1):
    def expected_numbers_label(expected_numbers_labels, numbers):
      return expected_numbers_labels if expected_numbers_labels is not None else np.concatenate(
        [one_hot(np.array(numbers) + 1, 10), np.array([[0.1] * 10 for i in range(5 - len(numbers))])])
    expected_numbers_labels = expected_numbers_label(expected_numbers_labels, [1, 9])
    self.assertNDArrayNear(nlb[0], expected_numbers_labels, 1e-5)
    expected_numbers_labels_1 = expected_numbers_label(expected_numbers_labels_1, [2, 3])
    self.assertNDArrayNear(nlb[1], expected_numbers_labels_1, 1e-5)

  def test_read_whole_train_metadata(self):
    max_number_length, batch_size, size = 5, 64, (64, 64)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_file_path = os.path.join(current_dir, '../data/train/metadata.pickle')
    data_dir_path = os.path.join(current_dir, '../data/train')
    metadata_handler = inputs.create_pickle_metadata_handler(metadata_file_path, max_number_length, data_dir_path)
    filenames, length_labels, numbers_labels = metadata_handler()


class DataReaderTest(tf.test.TestCase):

  @classmethod
  def createTestPickleMetadata(cls, test_data_count, test_dir_path=None):
    mat_metadata_file = DataReaderTest.createTestMatMetadata(test_data_count, test_dir_path)

    filenames, labels = [], []
    for i, data in enumerate(metadata_generator(mat_metadata_file)):
      filenames.append(data.filename)
      labels.append(data.label)

    metadata_file = os.path.join(test_dir_path, 'metadata.pickle')
    pickle.dump({'filenames': filenames, 'labels': labels}, open(metadata_file, 'wb'))

    return metadata_file

  @classmethod
  def createTestMatMetadata(cls, test_data_count, test_dir_path=None):
    test_dir_path = test_dir_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
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
    for i, data in enumerate(metadata_generator('./test_data/digitStruct.mat')):
      data_pack.append(data)
    self.assertEqual(data_pack[0].label, '19')
    self.assertEqual(data_pack[20].label, '2')
    self.assertEqual(data_pack[24].label, '601')
    self.assertEqual(data_pack[24].filename, '25.png')

  def test_metadata_generator(self):
    gen = metadata_generator('../data/train/digitStruct.mat')
    sampled = [gen.__next__() for i in range(30)]

    self.assertIsInstance(sampled[0], Data)
    self.assertEqual(sampled[0], Data('1.png', [ BBox('1', 77, 246, 81, 219), BBox('9', 81, 323, 96, 219) ]))
    self.assertEqual(sampled[20], Data('21.png', [ BBox(label=2, top=6.0, left=72.0, width=52.0, height=85.0) ]))