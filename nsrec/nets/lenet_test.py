import tensorflow as tf
from nsrec.nets import lenet
from nsrec.nets.test import BaseNetTest


class LenetTest(BaseNetTest):

  def test_cnn_layers_end_points(self):
    batch_size, height, width, channels = 5, 28, 28, 3
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with lenet.variable_scope([inputs]) as variable_scope:
        end_points_collection = lenet.end_points_collection_name(variable_scope)
        _, end_points = lenet.cnn_layers(inputs, variable_scope, end_points_collection)

      expected_names = [
        'lenet/conv1',
        'lenet/pool1',
        'lenet/conv2',
        'lenet/pool2',
      ]
      self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def test_fc_layers_end_points(self):
    layers_name_prefix = "length"
    batch_size, height, width, channels = 5, 28, 28, 3
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with lenet.variable_scope([inputs]) as variable_scope:
        end_points_collection = lenet.end_points_collection_name(variable_scope)
        net, _ = lenet.cnn_layers(inputs, variable_scope, end_points_collection)
        _, end_points = lenet.fc_layers(net, variable_scope, end_points_collection,
                                        num_classes=10, name_prefix=layers_name_prefix)

      expected_names = [
        'lenet/%s_fc3' % layers_name_prefix,
        'lenet/%s_dropout3' % layers_name_prefix,
        'lenet/%s_fc4' % layers_name_prefix,
      ]
      self.assertSetEqual(
        set(filter(lambda x: x.find('_fc') != -1 or x.find('_drop') != -1, end_points.keys())),
        set(expected_names))

  def test_network_output(self):
    batch_size, height, width, channels = 5, 28, 28, 3
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with lenet.variable_scope([inputs]) as variable_scope:
        end_points_collection = lenet.end_points_collection_name(variable_scope)
        net, _ = lenet.cnn_layers(inputs, variable_scope, end_points_collection)
        output, _ = lenet.fc_layers(net, variable_scope, end_points_collection, num_classes=10)

        self.checkOutputs({
          output: (5, 10),
        })


if __name__ == '__main__':
  tf.test.main()
