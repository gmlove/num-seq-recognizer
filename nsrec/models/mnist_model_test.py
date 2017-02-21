import tensorflow as tf
from nsrec.models.mnist_model import CNNMnistTrainModel
from nsrec.models.model_config import CNNGeneralModelConfig


class MnistModelTest(tf.test.TestCase):

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

