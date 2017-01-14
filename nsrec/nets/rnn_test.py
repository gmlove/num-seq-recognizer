import tensorflow as tf
from tensorflow.contrib import slim

from nsrec.nets import lenet, rnn
from nsrec.nets.test import BaseNetTest


class RNNTest(BaseNetTest):

  def test_network_output(self):
    batch_size, height, width, channels = 8, 28, 28, 3
    max_num_length, embedding_size = 5, 512
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, channels))
      with lenet.variable_scope([inputs]) as variable_scope:
        end_points_collection = lenet.end_points_collection_name(variable_scope)
        net, _ = lenet.cnn_layers(inputs, variable_scope, end_points_collection)
        net = slim.fully_connected(net, embedding_size, activation_fn=None, scope='fc0')
        net = rnn.rnn_layers(net, tf.ones((max_num_length, batch_size), dtype=tf.int32), embedding_size)
        net = tf.reshape(net, [-1, embedding_size])
        output = slim.fully_connected(net, 11, activation_fn=None, scope='fc4')

        self.checkOutputs({
          output: (batch_size * max_num_length, 11),
        })


if __name__ == '__main__':
  tf.test.main()
