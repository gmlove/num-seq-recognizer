import os

import tensorflow as tf
from nsrec.models.cnn_model import CNNNSRModelConfig
from nsrec.models.rnn_model import RNNTrainModel
from nsrec import inputs
from nsrec.inputs.inputs_test import DataReaderTest


class RNNModelTest(tf.test.TestCase):

  def test_train_rnn(self):
    metadata_file_path = DataReaderTest.getTestMetadata()
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
