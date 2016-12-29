import tensorflow as tf
from nsrec.nets import inception_v3
from nsrec.nets.test import BaseNetTest


class InceptionV3Test(BaseNetTest):

  def test_network_output(self):
    batch_size, height, width, channels = 5, inception_v3.image_height, inception_v3.image_width, 3
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with inception_v3.variable_scope([inputs]) as variable_scope:
        end_points_collection_name = inception_v3.end_points_collection_name(variable_scope)
        net, end_points_collection = inception_v3.cnn_layers(inputs, variable_scope, end_points_collection_name)
        output, end_points_collection = inception_v3.fc_layers(net, variable_scope, end_points_collection, num_classes=10)

        self.checkOutputs({
          output: (5, 10),
        })


if __name__ == '__main__':
  tf.test.main()
