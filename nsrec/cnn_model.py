import os
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from nsrec import inputs
from nsrec.nets import lenet, alexnet, inception_v3, iclr_mnr, lenet_v1, lenet_v2
from nsrec.np_ops import correct_count


class CNNGeneralModelConfig(object):

  def __init__(self, **kwargs):
    self.num_classes = None
    self.net_type = "lenet"
    self.force_size = None
    self.batch_size = 64

    for attr in ['num_classes', 'net_type', 'batch_size', 'force_size']:
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))

    self.final_cnn_net = None

  def __str__(self):
    return 'Config(%s)' % ', '.join(
      ['%s=%s'%(attr, getattr(self, attr))
       for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))])

  @property
  def cnn_net(self):
    if self.final_cnn_net:
      return self.final_cnn_net
    if self.net_type == 'alexnet':
      tf.logging.info('using alexnet')
      self.final_cnn_net = alexnet
      return alexnet
    elif self.net_type == 'inception_v3':
      tf.logging.info('using inception_v3 net')
      self.final_cnn_net = inception_v3
      return inception_v3
    elif self.net_type == 'iclr_mnr':
      tf.logging.info('using iclr_mnr net')
      self.final_cnn_net = iclr_mnr
      return iclr_mnr
    elif self.net_type == 'lenet_v1':
      tf.logging.info('using lenet_v1 net')
      self.final_cnn_net = lenet_v1
      return lenet_v1
    elif self.net_type == 'lenet_v2':
      tf.logging.info('using lenet_v2 net')
      self.final_cnn_net = lenet_v2
      return lenet_v2
    else:
      self.final_cnn_net = lenet
      tf.logging.info('using lenet')
      return lenet

  @property
  def size(self):
    if self.force_size is not None:
      return self.force_size
    return [self.cnn_net.image_height, self.cnn_net.image_width]


class CNNNSRModelConfig(CNNGeneralModelConfig):

  def __init__(self, **kwargs):
    super(CNNNSRModelConfig, self).__init__(**kwargs)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.data_dir_path = os.path.join(current_dir, '../data/train')
    self.metadata_file_path = os.path.join(self.data_dir_path, 'metadata.pickle')
    self.max_number_length = 5
    self.create_metadata_handler_fn = inputs.create_pickle_metadata_handler
    self.num_classes = self.max_number_length

    for attr in ['data_dir_path', 'metadata_file_path', 'max_number_length',
                 'create_metadata_handler_fn']:
      if kwargs.get(attr, None) is None:
        continue
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))


class CNNNSRInferModelConfig(CNNGeneralModelConfig):

  def __init__(self, **kwargs):
    super(CNNNSRInferModelConfig, self).__init__(**kwargs)

    self.max_number_length = 5
    self.images_to_infer = None

    for attr in ['max_number_length', 'images_to_infer']:
      if kwargs.get(attr, None) is None:
        continue
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))


class CNNGeneralModelBase:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.variables = None
    self.data_batches = None
    self.model_output = None
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
      loss = tf.nn.softmax_cross_entropy_with_logits(self.model_output, self.label_batches)
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
      metadata_handler = config.create_metadata_handler_fn(
        config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, self.label_batches, _ = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training)

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
      metadata_handler = config.create_metadata_handler_fn(
        config.metadata_file_path, config.max_number_length, config.data_dir_path)
      self.data_batches, self.length_label_batches, numbers_label_batches = \
        inputs.batches(metadata_handler, config.max_number_length, config.batch_size, config.size,
                       is_training=self.is_training)
      for i in range(self.max_number_length):
        self.numbers_label_batches.append(numbers_label_batches[:, i, :])

  def _setup_loss(self):
    number_losses = []
    with ops.name_scope(None, 'Loss') as sc:
      length_loss = tf.nn.softmax_cross_entropy_with_logits(self.length_output, self.length_label_batches)
      # tf.log will cause NaN issue
      # length_loss = tf.log(tf.reduce_mean(length_loss), 'length_loss')
      length_loss = tf.reduce_mean(length_loss, name='length_loss')
      self.total_loss = length_loss

      for i in range(self.max_number_length):
        number_loss = tf.nn.softmax_cross_entropy_with_logits(self.numbers_output[i], self.numbers_label_batches[i])
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



class CNNNSRInferenceModel(CNNNSRModelBase):

  def __init__(self, config):
    super(CNNNSRInferenceModel, self).__init__(config)
    self.length_pb = None
    self.numbers_pb = []
    self.is_training = False

  def _setup_input(self):
    self.data_batches = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3))

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
      feed_dict={self.data_batches: input_data})
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


def create_model(FLAGS, mode='train'):
  assert mode in ['train', 'eval', 'inference']

  model_clz = {
    'length-train': CNNLengthTrainModel,
    'length-eval': CNNNSREvalModel,
    'mnist-train': CNNMnistTrainModel,
    'all-train': CNNNSRTrainModel,
    'all-eval': CNNNSREvalModel,
    'all-inference': CNNNSRInferenceModel
  }

  key = '%s-%s' % (FLAGS.cnn_model_type, mode)
  if key not in model_clz:
    raise Exception('Unimplemented model: model_type=%s, mode=%s' % (FLAGS.cnn_model_type, mode))

  if FLAGS.cnn_model_type in ['length', 'all']:
    if mode in ['train', 'eval']:
      config = CNNNSRModelConfig(metadata_file_path=FLAGS.metadata_file_path,
                                 data_dir_path=FLAGS.data_dir_path,
                                 batch_size=FLAGS.batch_size,
                                 net_type=FLAGS.net_type,
                                 max_number_length=FLAGS.max_number_length)
    else:
      config = CNNNSRInferModelConfig(net_type=FLAGS.net_type,
                                      max_number_length=FLAGS.max_number_length)
  else:
    config = CNNGeneralModelConfig(batch_size=FLAGS.batch_size,
                                   net_type=FLAGS.net_type,
                                   num_classes=10)

  return model_clz[key](config)
