import os

import tensorflow as tf

import inputs
from cnn_model import CNNNSRModelConfig
from inputs_test import DataReaderTest
from rnn_model import RNNTrainModel


class RNNModelTest(tf.test.TestCase):

  def test_train_rnn(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestPickleMetadata(25, metadata_dir_path)
    config = CNNNSRModelConfig(metadata_file_path=metadata_file_path, batch_size=2,
                               create_metadata_handler_fn=inputs.create_pickle_metadata_handler)

    with self.test_session():
      model = RNNTrainModel(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)
