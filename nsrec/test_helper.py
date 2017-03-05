import os

import h5py

from nsrec.data_preprocessor import write_tf_records
from nsrec.inputs import metadata_generator


def relative_file(path):
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


test_data_count = 25
test_dir_path = relative_file('test_data')
train_mat_metadata_file = relative_file('../data/train/digitStruct.mat')
test_metadata_file = os.path.join(test_dir_path, 'metadata.pickle')
test_data_file = os.path.join(test_dir_path, 'data.tfrecords')
test_mat_metadata_file = os.path.join(test_dir_path, 'digitStruct.mat')
train_data_dir_path = relative_file('../data/train')
test_graph_file = os.path.join(test_dir_path, 'graph.pb')
output_graph_file = relative_file('../output/export/graph.pb')
output_bbox_graph_file = relative_file('../output/export/graph-bbox.pb')
_test_metadata_file_created = False


def get_test_metadata():
  global _test_metadata_file_created
  if not _test_metadata_file_created:
    _create_test_data()
    _test_metadata_file_created = True
  return test_data_file


def _create_test_data():
  mat_metadata_file = get_mat_test_metadata()

  filenames, labels, bboxes, sep_bboxes = [], [], [], []
  for i, data in enumerate(metadata_generator(mat_metadata_file)):
    filenames.append(data.filename)
    labels.append(data.label)
    bboxes.append(data.bbox())
    sep_bboxes.append([[int(bb.left), int(bb.top), int(bb.width), int(bb.height)] for bb in data.bboxes])

  write_tf_records(filenames, labels, 5, bboxes, sep_bboxes, 'full', train_data_dir_path, test_data_file)


def get_mat_test_metadata():
  test_f = h5py.File(test_mat_metadata_file, 'w')

  f = h5py.File(train_mat_metadata_file)
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
  return test_mat_metadata_file

