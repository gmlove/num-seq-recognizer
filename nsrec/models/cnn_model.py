import tensorflow as tf
from nsrec.models.nsr_bbox_model import CNNBBoxInferModel
from nsrec.models.nsr_model import CNNNSRInferenceModel
from nsrec.utils.ops import assign_vars, gray_scale, stack_output
from tensorflow.python.framework import ops


class CNNCombinedToExportModel:

  def __init__(self, config):
    self.config = config
    self.bbox_infer_model = CNNBBoxInferModel(config)
    self.infer_model = CNNNSRInferenceModel(config)
    self.bbox_vars_dict = None
    self.vars_dict = None

  def init(self, bbox_vars_dict, vars_dict):
    self.bbox_vars_dict = bbox_vars_dict
    self.vars_dict = vars_dict

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, 512, 512, 3), name='input')
    resized = tf.image.resize_images(self.inputs, self.config.size)
    self.data_batches = gray_scale(resized) if self.config.gray_scale else resized

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_net(self):
    with tf.variable_scope('bbox'):
      inputs_0 = tf.Variable(trainable=False, validate_shape=(None, self.config.size[0], self.config.size[1], 3))
      self.bbox_infer_model._setup_input(inputs_0)
      assign_op = tf.assign(inputs_0, self.data_batches)
      with tf.control_dependencies([assign_op]):
        self.bbox_infer_model._setup_net()

    def crop_bbox(width, height, input, bbox):
      expand_rate = 0.1
      top = tf.maximum(tf.floor(bbox[1] * height - height * expand_rate), 0)
      bottom = tf.minimum(tf.floor((bbox[1] + bbox[3]) * height + height * expand_rate), height)
      left = tf.maximum(tf.floor(bbox[0] * width - width * expand_rate), 0)
      right = tf.minimum((tf.floor(bbox[0] + bbox[2]) * width + width * expand_rate), width)
      top = tf.cond(top >= bottom, lambda: tf.identity(0), lambda: tf.identity(top))
      bottom = tf.cond(top >= bottom, lambda: tf.identity(height), lambda: tf.identity(bottom))
      left = tf.cond(left >= right, lambda: tf.identity(0), lambda: tf.identity(left))
      right = tf.cond(left >= right, lambda: tf.identity(width), lambda: tf.identity(right))
      return input[top:bottom, left:right, :]

    with tf.variable_scope('nsr'):
      origin_width, origin_height = 512, 512
      inputs_1 = tf.Variable(trainable=False, validate_shape=(None, self.config.size[0], self.config.size[1], 3))
      self.infer_model._setup_input(inputs_1)
      inputs = self.bbox_infer_model.inputs
      bboxes = self.bbox_infer_model.model_output
      inputs = tf.stack([crop_bbox(origin_width, origin_height, inputs[i], bboxes[i]) for i in range(self.config.batch_size)])
      inputs = tf.image.resize_images(inputs, self.config.size)
      assign_op = tf.assign(inputs_1, inputs)
      with tf.control_dependencies([assign_op]):
        self.infer_model._setup_net()

    vars_dict = self._vars()
    assign_ops = assign_vars(vars_dict, self.bbox_vars_dict, 'bbox')
    assign_ops.extend(assign_vars(vars_dict, self.vars_dict, 'nsr'))
    with tf.control_dependencies(assign_ops):
      self.output = stack_output(self.max_number_length, self.length_output, self.numbers_output)

  def _setup_global_step(self):
    pass

  def _setup_loss(self):
    pass

  def _setup_accuracy(self, **kwargs):
    pass


