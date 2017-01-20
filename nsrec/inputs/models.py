
def objeq(obj1, obj2, attr):
  if type(obj1) is not type(obj2):
    return False

  return all([getattr(obj1, name) == getattr(obj2, name) for name in attr])


def _objstr(obj):
  if isinstance(obj, (list, tuple)):
    to_format = '[%s]' if isinstance(obj, list) else '(%s)'
    return to_format % ', '.join(['%s' % i for i in obj])
  else:
    return '%s' % obj


def objstr(obj):
  clsname = obj.__class__.__name__
  attrs = obj.__class__.attrs
  return '%s(%s)' % (clsname, ', '.join(['%s=%s'%(attr, _objstr(getattr(obj, attr))) for attr in attrs]))


class ModelBase():
  attrs = []

  def __eq__(self, other):
    return objeq(self, other, self.__class__.attrs)

  def __str__(self):
    return objstr(self)


class BBox(ModelBase):

  attrs = 'label top left width height'.split()

  def __init__(self, label, top, left, width, height):
    self.label = str(int(label))
    self.label = '0' if self.label == '10' else self.label

    assert len(self.label) == 1

    self.left = left
    self.top = top
    self.height = height
    self.width = width


class Data(ModelBase):

  attrs = 'filename bboxes label'.split()

  def __init__(self, filename, bboxes):
    self.filename = filename
    self.bboxes = bboxes
    self.label = ''.join([bbox.label for bbox in bboxes])

  def bbox(self):
    min_left = min([bb.left for bb in self.bboxes])
    min_top = min([bb.top for bb in self.bboxes])
    max_right = max(bb.left + bb.width for bb in self.bboxes)
    max_bottom = max(bb.top + bb.height for bb in self.bboxes)
    return (int(min_left), int(min_top), int(max_right - min_left), int(max_bottom - min_top))

if __name__ == '__main__':
  assert Data('1', [BBox('1', 1, 1, 1, 1), BBox('2', 2, 2, 2, 2)]) == \
         Data('1', [BBox('1', 1, 1, 1, 1), BBox('2', 2, 2, 2, 2)])
  assert BBox('0', 1, 2, 3, 4).__str__() == \
         'BBox(label=0, top=1, left=2, width=3, height=4)'
  assert (Data('1', [BBox('1', 1, 1, 1, 1), BBox('2', 2, 2, 2, 2)]).__str__()) == \
         'Data(filename=1, bboxes=[BBox(label=1, top=1, left=1, width=1, height=1), BBox(label=2, top=2, left=2, width=2, height=2)], label=12)'
  assert Data('1', [BBox('1', 1, 1, 3, 4), BBox('2', 0, 2, 4, 2)]).bbox().__str__() ==  (1, 0, 5, 5).__str__()
