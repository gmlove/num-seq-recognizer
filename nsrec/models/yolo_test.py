import tensorflow as tf
import numpy as np

from nsrec.models.model_config import YOLOModelConfig, YOLOInferModelConfig
from nsrec import test_helper
from nsrec.models.yolo import YOLOTrainModel, YOLOEvalModel, YOLOInferModel


class YOLOModelTest(tf.test.TestCase):

  def test_train_model(self):
    with self.test_session():
      model = YOLOTrainModel(self.create_test_config())
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

  def create_test_config(self):
    data_file_path = test_helper.get_test_metadata()
    config = YOLOModelConfig(data_file_path=data_file_path, net_type='yolo', batch_size=2)
    return config

  def test_eval_model(self):
    with self.test_session() as sess:
      model = YOLOEvalModel(self.create_test_config())
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for i in range(2):
        print('batch %s correct count: %s' % (i, model.correct_count(sess)))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_inference(self):
    with self.test_session() as sess:
      model = YOLOInferModel(YOLOInferModelConfig(force_size=[416, 416], threshold=0.05))
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      labels = model.infer(sess, [np.ones((100, 100, 3)), np.ones((100, 100, 3))])
      print('infered labels for data: %s' % (labels, ))
