from functools import reduce

import numpy as np

import tensorflow as tf
from nsrec.models.model_config import CNNNSRModelConfig, CNNNSRInferModelConfig, CNNGeneralModelConfig
from nsrec import inputs
from tensorflow.python.framework import ops
from nsrec.utils.np_ops import correct_count


class CNNGeneralModelBase:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.model_output = None
    self.label_batches = None
    self.is_training = True

  def _setup_input(self):
    raise Exception('Implement me in subclass')

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection_name = self.cnn_net.end_points_collection_name(variable_scope)
      net, end_points_collection = self.cnn_net.cnn_layers(
        self.data_batches, variable_scope, end_points_collection_name)
      self.model_output, _ = self.cnn_net.fc_layers(
        net, variable_scope, end_points_collection,
        num_classes=self.config.num_classes, is_training=self.is_training, name_prefix='length')

  def _setup_loss(self):
    with ops.name_scope(None, 'Loss') as sc:
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model_output, labels=self.label_batches)
      loss = tf.reduce_mean(loss)
      self.total_loss = loss

    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  def _setup_accuracy(self, **kwargs):
    pass

  def _setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    tf.logging.info('using config: %s', self.config)
    self._setup_input()
    self._setup_net()
    self._setup_loss()
    self._setup_accuracy()
    self._setup_global_step()


class CNNMnistTrainModel(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNMnistTrainModel, self).__init__(config)

    self.label_batches = None
    self.total_loss = None
    self.global_step = None
    self.train_accuracy = None

  def _setup_input(self):
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches = \
        inputs.mnist_batches(self.config.batch_size, self.config.size, is_training=self.is_training)

  def _setup_accuracy(self):
    self.train_accuracy = softmax_accuracy(self.model_output, self.label_batches, 'accuracy/train')


class CNNBBoxTrainModel(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNBBoxTrainModel, self).__init__(config)

    self.config.num_classes = 4

    self.total_loss = None
    self.global_step = None
    self.train_accuracy = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches = \
        inputs.bbox_batches(config.data_file_path, config.batch_size, config.size,
                       is_training=self.is_training, channels=self.config.channels)

  def _setup_loss(self):
    with ops.name_scope(None, 'Loss') as sc:
      loss = tf.reduce_mean(tf.abs(self.label_batches - self.model_output))
      self.total_loss = loss

    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  def _setup_accuracy(self):
    pass


class CNNBBoxInferModel(CNNBBoxTrainModel):

  def __init__(self, config):
    super(CNNBBoxInferModel, self).__init__(config)

  def _setup_input(self, inputs=None):
    self.inputs = inputs or tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input-bbox')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_loss(self):
    pass

  def infer(self, sess, data):
    # image.shape: (height, width, 3)
    original_sizes = [image.shape for image in data]
    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    infered_bboxes  = sess.run(self.model_output, feed_dict={self.data_batches: input_data})
    result = []
    for i in range(len(data)):
      infered_bbox, original_size = infered_bboxes[i], original_sizes[i]
      result.append(list(map(lambda x: int(x), [
        infered_bbox[0] * original_size[1], infered_bbox[1] * original_size[0],
        infered_bbox[2] * original_size[1], infered_bbox[3] * original_size[0],
      ])))
    return result

  def vars(self, sess):
    return inference_vars(sess)


def softmax_accuracy(logits, label_batches, scope_name):
  with ops.name_scope(None, scope_name) as sc:
    correct_prediction = tf.equal(
      tf.argmax(tf.nn.softmax(logits), 1),
      tf.argmax(label_batches, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar(scope_name, accuracy)

    return accuracy


class CNNLengthTrainModel(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNLengthTrainModel, self).__init__(config)

    self.total_loss = None
    self.global_step = None
    self.train_accuracy = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.label_batches, _ = \
        inputs.batches(config.data_file_path, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training, channels=self.config.channels,
                       num_preprocess_threads=self.config.num_preprocess_threads)

  def _setup_accuracy(self):
    self.train_accuracy = softmax_accuracy(self.model_output, self.label_batches, 'accuracy/train')


class CNNNSRModelBase(CNNGeneralModelBase):

  def __init__(self, config):
    super(CNNNSRModelBase, self).__init__(config)
    self.length_output = None
    self.numbers_output = None
    self.max_number_length = self.config.max_number_length

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as variable_scope:
      end_points_collection_name = self.cnn_net.end_points_collection_name(variable_scope)
      net, end_points_collection = self.cnn_net.cnn_layers(self.data_batches, variable_scope, end_points_collection_name)
      self.length_output, _ = self.cnn_net.fc_layers(
        net, variable_scope, end_points_collection, dropout_keep_prob=0.9,
        num_classes=self.config.max_number_length, is_training=self.is_training, name_prefix='length')
      self.numbers_output = []
      for i in range(self.config.max_number_length):
        number_output, _ = self.cnn_net.fc_layers(
          net, variable_scope, end_points_collection, dropout_keep_prob=0.9,
          is_training=self.is_training, num_classes=11, name_prefix='number%s' % (i + 1))
        self.numbers_output.append(number_output)


class CNNNSRTrainModel(CNNNSRModelBase):

  def __init__(self, config):
    super(CNNNSRTrainModel, self).__init__(config)

    self.total_loss = None
    self.global_step = None
    self.batch_size = config.batch_size
    self.train_accuracy = None
    self.numbers_label_batches = []

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.length_label_batches, numbers_label_batches = \
        inputs.batches(config.data_file_path, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training, channels=self.config.channels,
                       num_preprocess_threads=self.config.num_preprocess_threads)
      for i in range(self.max_number_length):
        self.numbers_label_batches.append(numbers_label_batches[:, i, :])

  def _setup_loss(self):
    number_losses = []
    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.length_output, labels=self.length_label_batches)
      # tf.log will cause NaN issue
      # length_loss = tf.log(tf.reduce_mean(length_loss), 'length_loss')
      length_loss = tf.reduce_mean(length_loss, name='length_loss')
      self.total_loss = length_loss

      for i in range(self.max_number_length):
        number_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.numbers_output[i], labels=self.numbers_label_batches[i])
        # tf.log will cause NaN issue
        # number_loss = tf.log(tf.reduce_mean(number_loss), 'number%s_loss' % (i + 1))
        number_loss = tf.reduce_mean(number_loss, name='number%s_loss' % (i + 1))
        number_losses.append(number_loss)
        self.total_loss = self.total_loss + number_loss

    tf.summary.scalar("loss/length_loss", length_loss)
    for i in range(self.max_number_length):
      tf.summary.scalar("loss/number%s_loss" % (i + 1), number_losses[i])
    tf.summary.scalar("loss/total_loss", self.total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  def _setup_accuracy(self, op_name='accuracy/train'):
    softmax_accuracy(self.length_output, self.length_label_batches, 'accuracy/train/length')
    for i in range(self.max_number_length):
      softmax_accuracy(self.numbers_output[i], self.numbers_label_batches[i], 'accuracy/train/number%s' % (i + 1))


class CNNNSREvalModel(CNNNSRTrainModel):

  def __init__(self, config):
    super(CNNNSREvalModel, self).__init__(config)
    self.is_training = False

  def _setup_accuracy(self, op_name='accuracy/eval'):
    self.length_label_batches_pd = tf.nn.softmax(self.length_output)
    self.numbers_label_batches_pd = [None] * self.max_number_length
    for i in range(self.max_number_length):
      self.numbers_label_batches_pd[i] = tf.nn.softmax(self.numbers_output[i])

  def _setup_loss(self):
    pass

  def correct_count(self, sess):
    calculated_values = sess.run({
      'length_label_batches': self.length_label_batches,
      'numbers_label_batches': self.numbers_label_batches,
      'length_label_batches_pd': self.length_label_batches_pd,
      'numbers_label_batches_pd': self.numbers_label_batches_pd
    })
    length_label_batches, numbers_label_batches, \
    length_label_batches_pd, numbers_label_batches_pd = \
      calculated_values['length_label_batches'], calculated_values['numbers_label_batches'], \
      calculated_values['length_label_batches_pd'], calculated_values['numbers_label_batches_pd']

    return correct_count(length_label_batches, numbers_label_batches,
                         length_label_batches_pd, numbers_label_batches_pd)


def gray_scale(inputs):
  data_batches = tf.image.rgb_to_grayscale(inputs)
  shape = data_batches.get_shape().as_list()
  shape[-1] = 1
  data_batches.set_shape(shape)
  return data_batches


def inference_vars(sess):
  coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
  model_vars = sess.run(coll)
  model_vars_dict = dict(zip([v.name for v in coll], model_vars))
  return model_vars_dict


class CNNNSRInferenceModel(CNNNSRModelBase):

  def __init__(self, config):
    super(CNNNSRInferenceModel, self).__init__(config)
    self.length_pb = None
    self.numbers_pb = []
    self.is_training = False
    self.inputs = None

  def _setup_input(self, inputs=None):
    self.inputs = inputs or tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_loss(self):
    pass

  def _setup_net(self):
    super(CNNNSRInferenceModel, self)._setup_net()
    self.length_pb = tf.nn.softmax(self.length_output)
    for i in range(self.max_number_length):
      self.numbers_pb.append(tf.nn.softmax(self.numbers_output[i]))

  def infer(self, sess, data):
    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    length_pb, numbers_pb = sess.run(
      [self.length_pb, self.numbers_pb],
      feed_dict={self.inputs: input_data})
    length = np.argmax(length_pb, axis=1)
    numbers = np.argmax(numbers_pb, axis=2)

    labels = []
    length_prob = lambda i: length_pb[i][length[i]]
    number_prob = lambda i, j: numbers_pb[j][i][numbers[j][i]]

    for i in range(len(length)):
      label = ''.join([str(n) for n in numbers[:length[i] + 1, i]])
      probabilities = [length_prob(i)] + [number_prob(i, j) for j in range(length[i] + 1)]
      reduced_prob = reduce(lambda x, y: x * y, probabilities)
      labels.append((label, probabilities, reduced_prob))
    return labels

  def vars(self, sess):
    return inference_vars(sess)


def vars_assign_ops(vars_dict, saved_vars_dict, prefix=''):
  assign_ops = []
  for name, input_var in saved_vars_dict.items():
    assign_ops.append(tf.assign(vars_dict[prefix + name], input_var))
  return assign_ops


def stacked_output_ops(max_number_length, length_output, numbers_output, name='output'):
  length_pb = tf.nn.softmax(length_output)
  to_concat = [tf.reshape(length_pb, (max_number_length, ))]
  for i in range(max_number_length):
    to_concat.append(tf.reshape(tf.nn.softmax(numbers_output[i]), (11, )))
  return tf.concat(axis=0, values=to_concat, name=name)


class CNNNSRToExportModel(CNNNSRInferenceModel):

  def __init__(self, config):
    super(CNNNSRToExportModel, self).__init__(config)
    self.saved_vars_dict = None
    self.output = None

  def init(self, saved_vars_dict):
    self.saved_vars_dict = saved_vars_dict

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_net(self):
    super(CNNNSRToExportModel, self)._setup_net()

    assign_ops = vars_assign_ops(self._vars(), self.saved_vars_dict)
    with tf.control_dependencies(assign_ops):
      self.output = stacked_output_ops(
        self.max_number_length, self.length_output, self.numbers_output, name='output')


class CNNBboxToExportModel(CNNBBoxInferModel):

  def __init__(self, config):
    super(CNNBboxToExportModel, self).__init__(config)
    self.saved_vars_dict = None
    self.output = None

  def init(self, saved_vars_dict):
    self.saved_vars_dict = saved_vars_dict

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_net(self):
    super(CNNBBoxInferModel, self)._setup_net()

    assign_ops = vars_assign_ops(self._vars(), self.saved_vars_dict)
    with tf.control_dependencies(assign_ops):
      self.output = tf.reshape(self.model_output, (-1, ), 'output-bbox')


class CNNCombinedToExportModel(CNNGeneralModelBase):

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
    assign_ops = vars_assign_ops(vars_dict, self.bbox_vars_dict, 'bbox')
    assign_ops.extend(vars_assign_ops(vars_dict, self.vars_dict, 'nsr'))
    with tf.control_dependencies(assign_ops):
      self.output = stacked_output_ops(self.max_number_length, self.length_output, self.numbers_output)

  def _setup_global_step(self):
    pass

  def _setup_loss(self):
    pass

  def _setup_accuracy(self, **kwargs):
    pass


def create_model(FLAGS, mode='train'):
  assert mode in ['train', 'eval', 'inference', 'to_export']

  from models.rnn_model import RNNTrainModel, RNNEvalModel
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
    'batch_size': FLAGS.batch_size,
    'net_type': FLAGS.net_type,
    'gray_scale': FLAGS.gray_scale
  }

  if FLAGS.cnn_model_type in ['length', 'all', 'bbox']:
    params_dict.update({'max_number_length': FLAGS.max_number_length})
    if mode in ['train', 'eval']:
      params_dict.update({
        'num_preprocess_threads': FLAGS.num_preprocess_threads,
        'data_file_path': FLAGS.data_file_path
      })
      config = CNNNSRModelConfig(**params_dict)
    else:
      config = CNNNSRInferModelConfig(**params_dict)
  else:
    params_dict.update({'num_classes': 10})
    config = CNNGeneralModelConfig(**params_dict)

  return model_clz[key](config)
