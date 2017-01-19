import os

import tensorflow as tf

from nsrec.cnn_model import *
from nsrec.inputs_test import DataReaderTest


class CNNModelTest(tf.test.TestCase):

  def test_train_mnist_model(self):
    config = CNNGeneralModelConfig(batch_size=2, num_classes=10)
    with self.test_session():
      model = CNNMnistTrainModel(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

  def test_train_model_with_pickle_metadata(self):
    self.run_test(DataReaderTest.createTestPickleMetadata, inputs.create_pickle_metadata_handler)

  def test_train_length_model_with_pickle_metadata(self):
    self.run_test(DataReaderTest.createTestPickleMetadata, inputs.create_pickle_metadata_handler,
                  CNNLengthTrainModel)

  def test_train_bbox_model_with_pickle_metadata(self):
    self.run_test(DataReaderTest.createTestPickleMetadata, inputs.create_pickle_metadata_handler,
                  CNNBBoxTrainModel)

  def test_train_model_with_mat_metadata(self):
    self.run_test(DataReaderTest.createTestMatMetadata, inputs.create_mat_metadata_handler)

  def run_test(self, create_metadata_fn, create_metadata_handler_fn, model_cls=CNNNSRTrainModel):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = create_metadata_fn(25, metadata_dir_path)
    config = CNNNSRModelConfig(metadata_file_path=metadata_file_path, batch_size=2,
                               create_metadata_handler_fn=create_metadata_handler_fn)

    with self.test_session():
      model = model_cls(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

  def test_evaluation_correct_count(self):
    metadata_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    metadata_file_path = DataReaderTest.createTestPickleMetadata(25, metadata_dir_path)
    config = CNNNSRModelConfig(metadata_file_path=metadata_file_path, batch_size=2,
                               create_metadata_handler_fn=inputs.create_pickle_metadata_handler)

    with self.test_session() as sess:
      model = CNNNSREvalModel(config)
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for i in range(10):
        print('batch %s correct count: %s' % (i, model.correct_count(sess)))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_bbox_inference(self):
    self._test_inference(CNNNSRInferModelConfig(), CNNBBoxInferModel)

  def test_inference(self):
    self._test_inference(CNNNSRInferModelConfig(), CNNNSRInferenceModel)

  def _test_inference(self, config, model_cls):
    with self.test_session() as sess:
      model = model_cls(config)
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      labels = model.infer(sess, [np.ones((100, 100, 3)), np.ones((100, 100, 3))])
      print('infered labels for data: %s' % (labels, ))

  def test_export(self):
    config = CNNNSRInferModelConfig()

    with self.test_session() as sess:
      model = CNNNSRInferenceModel(config)
      model.build()
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      model_vars = model.vars(sess)

    with self.test_session() as sess:
      model = CNNNSRToExportModel(config)
      model.init(model_vars)
      model.build()
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      pbs = model.infer(sess, [np.ones((100, 100, 3))])
      print(pbs)
      self.assertEqual(len(pbs), 5 + 5 * 11)

if __name__ == '__main__':
  tf.test.main()

