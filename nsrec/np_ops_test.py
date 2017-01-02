import numpy as np
import tensorflow as tf

from nsrec.np_ops import one_hot, correct_count


class OpsTest(tf.test.TestCase):

  def test_one_hot(self):
    self.assertAllEqual(one_hot(3, 5), np.array([0, 0, 1, 0, 0], dtype=np.float32))

  def test_correct_count(self):
    self.assertEqual(correct_count(
      np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32), # 2, 1(max_number_length=3)
      [np.array([one_hot(3, 10), one_hot(7, 10), one_hot(6, 10)], dtype=np.float32),
       np.array([one_hot(1, 10), [0.1] * 10, one_hot(10, 10)], dtype=np.float32),
       np.array([one_hot(5, 10), [0.1] * 10, one_hot(2, 10)], dtype=np.float32)], # 204, 6, 591,
      np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), # 2, 1(max_number_length=3)
      [np.array([one_hot(3, 10), one_hot(8, 10), one_hot(6, 10)], dtype=np.float32),
       np.array([one_hot(1, 10), [0.1] * 10, one_hot(10, 10)], dtype=np.float32),
       np.array([one_hot(5, 10), [0.1] * 10, [0.1] * 10], dtype=np.float32)], # predicted: 204, 7, 59,
    ), 1)


if __name__ == '__main__':
  tf.test.main()
