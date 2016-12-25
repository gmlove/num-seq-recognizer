import tensorflow as tf
from nsrec.data_reader import metadata_generator
from nsrec.models import Data, BBox

class DataReaderTest(tf.test.TestCase):

  def test_meta_data_generator(self):
    gen = metadata_generator('../data/train/digitStruct.mat')
    sampled = [gen.__next__() for i in range(30)]
    self.assertIsInstance(sampled[0], Data)
    self.assertEqual(sampled[0], Data('1.png', [ BBox('1', 77, 246, 81, 219), BBox('9', 81, 323, 96, 219) ]))
    self.assertEqual(sampled[20], Data('21.png', [ BBox(label=2, top=6.0, left=72.0, width=52.0, height=85.0) ]))

if __name__ == "__main__":
  tf.test.main()
