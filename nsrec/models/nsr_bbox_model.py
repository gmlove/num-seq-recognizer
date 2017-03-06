import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import basic_net
from nsrec.utils.ops import global_step_variable, gray_scale, all_model_variables_data, assign_vars


class CNNBBoxTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.config.num_classes = 4

    self.total_loss = None
    self.global_step = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches = \
        inputs.bbox_batches(config.data_file_path, config.batch_size, config.size, config.max_number_length,
                            channels=self.config.channels)

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, True)

  def _setup_loss(self):
    with ops.name_scope(None, 'Loss') as sc:
      loss = tf.reduce_mean(tf.square(self.label_batches - self.model_output))
      self.total_loss = loss

    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_loss()
    self._setup_global_step()


class CNNBBoxInferModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.config.num_classes = 4

    self.model_output = None

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3))
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, False)

  def infer(self, sess, data):
    """
    Args:
      sess: tensorflow session
      data: raw image data, shape should be [count, height, width, 3]

    Returns:
      A list of inferred bbox in pixels, structured by (left, top, width, height).

    """
    # image.shape: (height, width, 3)
    original_sizes = [image.shape for image in data]
    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    inferred_bboxes = sess.run(self.model_output, feed_dict={self.data_batches: input_data})
    result = []
    for i in range(len(data)):
      inferred_bbox, original_size = inferred_bboxes[i], original_sizes[i]
      result.append(list(map(lambda x: int(x), [
        inferred_bbox[0] * original_size[1] / 100, inferred_bbox[1] * original_size[0] / 100,
        inferred_bbox[2] * original_size[1] / 100, inferred_bbox[3] * original_size[0] / 100,
      ])))
    return result

  def build(self):
    self._setup_input()
    self._setup_net()

  def vars(self, sess):
    return all_model_variables_data(sess)


class CNNBBoxToExportModel:
  INITIALIZER_NODE_NAME = "initializer-bbox"
  OUTPUT_NODE_NAME = 'output-bbox'
  INPUT_NODE_NAME = 'input-bbox'

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.config.num_classes = 4

    self.output = None
    self.initializer = None

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_input(self):
    self.inputs = tf.placeholder(
      tf.float32, (None, self.config.size[0], self.config.size[1], 3), name=self.INPUT_NODE_NAME)
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self, saved_vars_dict):
    model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, False)
    self.output = tf.reshape(model_output, (-1, ), self.OUTPUT_NODE_NAME)
    assign_ops = assign_vars(self._vars(), saved_vars_dict)
    self.initializer = tf.group(*assign_ops, name=self.INITIALIZER_NODE_NAME)

  def build(self, saved_vars_dict):
    self._setup_input()
    self._setup_net(saved_vars_dict)