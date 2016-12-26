import tensorflow as tf


class BaseNetTest(tf.test.TestCase):

  def checkOutputs(self, expected_shapes, feed_dict=None):
    """Verifies that the model produces expected outputs.

    Args:
      expected_shapes: A dict mapping Tensor or Tensor name to expected output
        shape.
      feed_dict: Values of Tensors to feed into Session.run().
    """
    fetches = list(expected_shapes.keys())

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(fetches, feed_dict)

    for index, output in enumerate(outputs):
      tensor = fetches[index]
      expected = expected_shapes[tensor]
      actual = output.shape
      if expected != actual:
        self.fail("Tensor %s has shape %s (expected %s)." %
                  (tensor, actual, expected))