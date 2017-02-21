import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import basic_net
from nsrec.models.mnist_model import CNNMnistTrainModel
from nsrec.models.model_config import CNNNSRModelConfig, CNNNSRInferModelConfig, CNNGeneralModelConfig
from nsrec.models.nsr_model import CNNNSRTrainModel, CNNNSREvalModel, CNNNSRInferenceModel, CNNNSRToExportModel, \
  RNNTrainModel, RNNEvalModel
from nsrec.utils.ops import all_model_variables_data, assign_vars, softmax_accuracy, gray_scale, stack_output, \
  global_step_variable, softmax_cross_entrophy_loss


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
        inputs.bbox_batches(config.data_file_path, config.batch_size, config.size, channels=self.config.channels)

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, True)

  def _setup_loss(self):
    with ops.name_scope(None, 'Loss') as sc:
      loss = tf.reduce_mean(tf.abs(self.label_batches - self.model_output))
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
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input-bbox')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, False)

  def infer(self, sess, data):
    # image.shape: (height, width, 3)
    original_sizes = [image.shape for image in data]
    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    inferred_bboxes = sess.run(self.model_output, feed_dict={self.data_batches: input_data})
    result = []
    for i in range(len(data)):
      inferred_bbox, original_size = inferred_bboxes[i], original_sizes[i]
      result.append(list(map(lambda x: int(x), [
        inferred_bbox[0] * original_size[1], inferred_bbox[1] * original_size[0],
        inferred_bbox[2] * original_size[1], inferred_bbox[3] * original_size[0],
      ])))
    return result

  def build(self):
    self._setup_input()
    self._setup_net()

  def vars(self, sess):
    return all_model_variables_data(sess)


class CNNLengthTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net

    self.data_batches = None
    self.label_batches = None

    self.total_loss = None
    self.global_step = None
    self.train_accuracy = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches, _ = \
        inputs.batches(config.data_file_path, config.max_number_length, config.batch_size, config.size,
                       is_training=True, channels=self.config.channels,
                       num_preprocess_threads=self.config.num_preprocess_threads)

  def _setup_net(self):
    self.model_output = basic_net(self.cnn_net, self.data_batches, self.config.max_number_length, True)

  def _setup_accuracy(self):
    self.train_accuracy = softmax_accuracy(self.model_output, self.label_batches, 'accuracy/train')

  def _setup_loss(self):
    self.total_loss = softmax_cross_entrophy_loss(self.model_output, self.label_batches)

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_accuracy()
    self._setup_loss()
    self._setup_global_step()


class CNNBboxToExportModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.config.num_classes = 4

    self.output = None

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input-bbox')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self, saved_vars_dict):
    model_output = basic_net(self.cnn_net, self.data_batches, self.config.num_classes, False)
    assign_ops = assign_vars(self._vars(), saved_vars_dict)
    with tf.control_dependencies(assign_ops):
      self.output = tf.reshape(model_output, (-1, ), 'output-bbox')

  def build(self, saved_vars_dict):
    self._setup_input()
    self._setup_net(saved_vars_dict)


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


def create_model(FLAGS, mode='train'):
  assert mode in ['train', 'eval', 'inference', 'to_export']

  model_clz = {
    'bbox-train': CNNBBoxTrainModel,
    'bbox-inference': CNNBBoxInferModel,
    'bbox-to_export': CNNBboxToExportModel,
    'length-train': CNNLengthTrainModel,
    'length-eval': CNNNSREvalModel,
    'mnist-train': CNNMnistTrainModel,
    'all-train': CNNNSRTrainModel,
    'all-train-rnn': RNNTrainModel,
    'all-eval': CNNNSREvalModel,
    'all-eval-rnn': RNNEvalModel,
    'all-inference': CNNNSRInferenceModel,
    'all-to_export': CNNNSRToExportModel
  }

  key = '%s-%s' % (FLAGS.cnn_model_type, mode)
  if FLAGS.rnn:
    key = key + '-rnn'
    tf.logging.info('using rnn')

  if key not in model_clz:
    raise Exception('Unimplemented model: model_type=%s, mode=%s' % (FLAGS.cnn_model_type, mode))

  params_dict = {
    'net_type': FLAGS.net_type,
    'gray_scale': FLAGS.gray_scale
  }

  if FLAGS.cnn_model_type in ['length', 'all', 'bbox']:
    params_dict.update({'max_number_length': FLAGS.max_number_length})
    if mode in ['train', 'eval']:
      params_dict.update({
        'num_preprocess_threads': FLAGS.num_preprocess_threads,
        'data_file_path': FLAGS.data_file_path,
        'batch_size': FLAGS.batch_size,
      })
      config = CNNNSRModelConfig(**params_dict)
    else:
      config = CNNNSRInferModelConfig(**params_dict)
  else:
    params_dict.update({
      'num_classes': 10,
      'batch_size': FLAGS.batch_size,
    })
    config = CNNGeneralModelConfig(**params_dict)

  return model_clz[key](config)
