from nsrec import test_helper
from nsrec.models.cnn_model import *


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

  def test_train_model(self):
    self._run_test()

  def test_train_length_model(self):
    self._run_test(CNNLengthTrainModel)

  def test_train_bbox_model(self):
    self._run_test(CNNBBoxTrainModel)

  def _run_test(self, model_cls=CNNNSRTrainModel):
    data_file_path = test_helper.get_test_metadata()
    config = CNNNSRModelConfig(data_file_path=data_file_path, batch_size=2)

    with self.test_session():
      model = model_cls(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

  def test_evaluation_correct_count(self):
    data_file_path = test_helper.get_test_metadata()
    config = CNNNSRModelConfig(data_file_path=data_file_path, batch_size=2)

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

  def test_model_export(self):
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
      pbs = sess.run(model.output, feed_dict={model.inputs: np.ones((1, 64, 64, 3))})
      print(pbs)
      self.assertEqual(len(pbs), 5 + 5 * 11)

  def test_bbox_model_export(self):
    config = CNNNSRInferModelConfig()

    with self.test_session() as sess:
      model = CNNBBoxInferModel(config)
      model.build()
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      model_vars = model.vars(sess)

    with self.test_session() as sess:
      model = CNNBboxToExportModel(config)
      model.init(model_vars)
      model.build()
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      pbs = sess.run(model.output, feed_dict={model.inputs: np.ones((1, config.size[0], config.size[1], 3))})
      print(pbs)
      self.assertEqual(len(pbs), 4)

if __name__ == '__main__':
  tf.test.main()

