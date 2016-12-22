import tensorflow as tf
from nsrec.data_reader import metadata_generator
from nsrec.models import Data, BBox

class DataReaderTest(tf.test.TestCase):

  def test_meta_data_generator(self):
    gen = metadata_generator('../data/train/digitStruct.mat')
    first_data = gen.__next__()
    self.assertIsInstance(first_data, Data)
    print(first_data)
    self.assertEqual(first_data, Data('1.png', [ BBox('1', 77, 246, 81, 219), BBox('9', 81, 323, 96, 219) ]))


if __name__ == "__main__":
  tf.test.main()
