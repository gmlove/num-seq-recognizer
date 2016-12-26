import tensorflow as tf

from nsrec.cnn_model import *
from nsrec.data_preprocessor_test import DataReaderTest


class CNNModelTest(tf.test.TestCase):

  def test_train_model_with_pickle_metadata(self):
    self.run_test(DataReaderTest.createTestPickleMetadata, inputs.create_pickle_metadata_handler)

  def test_train_model_with_mat_metadata(self):
    self.run_test(DataReaderTest.createTestMatMetadata, inputs.create_mat_metadata_handler)

  def run_test(self, create_metadata_fn, create_metadata_handler_fn):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = create_metadata_fn(25, metadata_dir_path)
    config = CNNModelConfig(metadata_file_path=metadata_file_path, batch_size=2,
                            create_metadata_handler_fn=create_metadata_handler_fn)

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

