import tensorflow as tf

from nsrec.cnn_model import *
from nsrec.data_reader_test import DataReaderTest


class CNNModelTest(tf.test.TestCase):

  def test_train_model(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestData(25, metadata_dir_path)
    config = CNNModelConfig(metadata_file_path=metadata_file_path, batch_size=2)

    with self.test_session():
      model = CNNTrainModel(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

if __name__ == '__main__':
  tf.test.main()

