import h5py

from nsrec.models import BBox, Data


def metadata_generator(file_path):
  f = h5py.File(file_path)
  refs, ds = f['#refs#'], f['digitStruct']

  def bboxes(i):
    attr_names = 'label top left width height'.split()
    bboxes = []
    bboxes_raw = refs[ds['bbox'][i][0]]
    bboxes_count = bboxes_raw['label'].value.shape[0]
    for j in range(bboxes_count):
      attr_value = lambda attr_name: refs[bboxes_raw[attr_name].value[j][0]].value.reshape(-1)[0]
      bboxes.append(BBox(*[attr_value(an) for an in attr_names]))
    return bboxes

  for i, name in enumerate(ds['name']):
    ords = refs[name[0]].value
    name_str = ''.join([chr(ord) for ord in ords.reshape(-1)])
    yield Data(name_str, bboxes(i))

