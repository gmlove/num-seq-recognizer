import os

import h5py
import tensorflow as tf
from six.moves import cPickle as pickle
from nsrec.data_preprocessor import metadata_generator
from nsrec.models import Data, BBox

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

if __name__ == "__main__":
  tf.test.main()
