
def objeq(obj1, obj2, attr):
  if type(obj1) is not type(obj2):
    return False

  return all([getattr(obj1, name) == getattr(obj2, name) for name in attr])

def objstr(obj):
  clsname = obj.__class__.__name__
  attrs = obj.__class__.attrs
  return '%s(%s)' % (clsname, ', '.join(['%s=%s'%(attr, getattr(obj, attr)) for attr in attrs]))

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

if __name__ == '__main__':
  assert Data('1', [BBox('1', 1, 1, 1, 1), BBox('2', 2, 2, 2, 2)]) == Data('1', [BBox('1', 1, 1, 1, 1), BBox('2', 2, 2, 2, 2)])
  assert BBox('0', 1, 2, 3, 4).__str__() == 'BBox(label=0, top=1, left=2, width=3, height=4)'