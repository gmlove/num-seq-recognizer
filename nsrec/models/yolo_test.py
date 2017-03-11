import tensorflow as tf
import numpy as np
from nsrec.models import CNNNSRInferModelConfig
from nsrec import test_helper
from nsrec.models import CNNNSRModelConfig
from nsrec.models.yolo import YOLOTrainModel, YOLOEvalModel, YOLOInferModel


class YOLOTrainModelTest(tf.test.TestCase):

  def test_train_model(self):
    data_file_path = test_helper.get_test_metadata()
    config = CNNNSRModelConfig(data_file_path=data_file_path, batch_size=2, force_size=[416, 416])

    with self.test_session():
      model = YOLOTrainModel(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step,
        learning_rate=0.1, optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(
        train_op, None, number_of_steps=2)

  def test_eval_model(self):
    data_file_path = test_helper.get_test_metadata()
    config = CNNNSRModelConfig(data_file_path=data_file_path, batch_size=2, force_size=[416, 416])

    with self.test_session() as sess:
      model = YOLOEvalModel(config)
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
      model = YOLOInferModel(CNNNSRInferModelConfig(force_size=[416, 416]))
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      labels = model.infer(sess, [np.ones((100, 100, 3)), np.ones((100, 100, 3))])
      print('infered labels for data: %s' % (labels, ))
