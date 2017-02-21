from nsrec import test_helper
from models.nsr_length_model import *
from models.model_config import *


class NSRLengthModelTest(tf.test.TestCase):

  def test_train_length_model(self):
    data_file_path = test_helper.get_test_metadata()
    config = CNNNSRModelConfig(data_file_path=data_file_path, batch_size=2)

    with self.test_session():
      model = CNNLengthTrainModel(config)
      model.build()

      train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss, global_step=model.global_step, learning_rate=0.1,
        optimizer=tf.train.MomentumOptimizer(0.5, momentum=0.5))
      tf.contrib.slim.learning.train(train_op, None, number_of_steps=2)
