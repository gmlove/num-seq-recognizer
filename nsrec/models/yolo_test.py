import tensorflow as tf
import numpy as np

from nsrec.models.model_config import YOLOModelConfig, YOLOInferModelConfig
from nsrec import test_helper
from nsrec.models.yolo import YOLOTrainModel, YOLOEvalModel, YOLOInferModel, extract_label_as_array, build_export_output


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
      model = YOLOInferModel(YOLOInferModelConfig(force_size=[104, 104], net_type='simple_yolo', threshold=0.05))
      model.build()

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      labels = model.infer(sess, [np.ones((100, 100, 3)), np.ones((100, 100, 3))])
      print('infered labels for data: %s' % (labels, ))

  def test_build_export_output(self):
    H, W, B, C, threshold = 2, 2, 5, 2, 0.1
    net_out = np.random.normal(0.3, 0.3, (H, W, B * (5 + C)))
    _lefts, _boxes, _classes, _probs = extract_label_as_array(net_out, H, W, B, C, threshold)

    with self.test_session() as sess:
      lefts, boxes, classes, probs = sess.run(build_export_output(net_out, H, W, B, C, threshold))

      self.assertEqual(lefts.shape, _lefts.shape)
      self.assertAllClose(lefts, _lefts)
      self.assertAllClose([b for b in boxes if b[0] > 0], [b for b in _boxes if b[0] > 0])
      self.assertAllClose([c for i, c in enumerate(classes) if boxes[i][0] > 0],
                          [c for i, c in enumerate(_classes) if _boxes[i][0] > 0],)
      self.assertAllClose([c for i, c in enumerate(probs) if boxes[i][0] > 0],
                          [c for i, c in enumerate(_probs) if _boxes[i][0] > 0],)
