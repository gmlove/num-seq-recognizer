import tensorflow as tf

from nsrec.nets import simple_yolo as net_builder
from nsrec.nets.test import BaseNetTest


class SimpleYoloTest(BaseNetTest):

  def test_network_output(self):
    batch_size, height, width, channels = 5, net_builder.image_height, net_builder.image_width, 3
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with net_builder.variable_scope([inputs]) as variable_scope:
        end_points_collection = net_builder.end_points_collection_name(variable_scope)
        net, _ = net_builder.cnn_layers(inputs, variable_scope, end_points_collection)

        self.checkOutputs({
          net: (batch_size, 13, 13, 75),
        })

  def test_replace_extract_image_patches(self):
    with self.test_session() as sess:
      net = tf.random_uniform([2, 26, 26, 128])
      net1 = tf.extract_image_patches(
        net, [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')
      print(net.get_shape().as_list())
      from tensorflow.contrib import slim
      net2 = slim.conv2d(net, 512, [1, 1], 2)
      print(net.get_shape().as_list())



if __name__ == '__main__':
  tf.test.main()
