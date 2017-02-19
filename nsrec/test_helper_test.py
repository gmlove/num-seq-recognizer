import tensorflow as tf
from nsrec import test_helper
from nsrec.inputs import metadata_generator, Data, BBox


class TestHelperTest(tf.test.TestCase):

  def test_create_metadata_file_for_testing(self):
    test_helper.get_mat_test_metadata()

    data_pack = []
    for i, data in enumerate(metadata_generator(test_helper.test_mat_metadata_file)):
      data_pack.append(data)
    self.assertEqual(data_pack[0].label, '19')
    self.assertEqual(data_pack[20].label, '2')
    self.assertEqual(data_pack[24].label, '601')
    self.assertEqual(data_pack[24].filename, '25.png')

  def test_metadata_generator(self):
    gen = metadata_generator(test_helper.train_mat_metadata_file)
    sampled = [gen.__next__() for i in range(30)]

    self.assertIsInstance(sampled[0], Data)
    self.assertEqual(sampled[0], Data('1.png', [BBox('1', 77, 246, 81, 219), BBox('9', 81, 323, 96, 219)]))
    self.assertEqual(sampled[20], Data('21.png', [BBox(label=2, top=6.0, left=72.0, width=52.0, height=85.0)]))
