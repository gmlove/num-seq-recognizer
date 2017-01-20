import os

import inputs
import tensorflow as tf
from nsrec.nets import lenet, alexnet, inception_v3, iclr_mnr, lenet_v2, lenet_v1


class CNNGeneralModelConfig(object):

  def __init__(self, **kwargs):
    self.num_classes = None
    self.net_type = "lenet"
    self.force_size = None
    self.batch_size = 64
    self.gray_scale = False

    for attr in ['num_classes', 'net_type', 'batch_size', 'force_size', 'gray_scale']:
      setattr(self, attr, kwargs.get(attr, getattr(self, attr)))

    self._final_cnn_net = None

  def __str__(self):
    return 'Config(%s)' % ', '.join(
      ['%s=%s'%(attr, getattr(self, attr))
       for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))])

  @property
  def channels(self):
    return 1 if self.gray_scale else 3

  @property
  def cnn_net(self):
    net_dict = {
      'alexnet': alexnet,
      'inception_v3': inception_v3,
      'iclr_mnr': iclr_mnr,
      'lenet': lenet,
      'lenet_v1': lenet_v1,
      'lenet_v2': lenet_v2,
    }
    if self._final_cnn_net:
      return self._final_cnn_net
    if self.net_type in net_dict:
      tf.logging.info('using %s', self.net_type)
      self._final_cnn_net = net_dict[self.net_type]
      return net_dict[self.net_type]
    else:
      raise Exception('Unsupported net: %s' % self.net_type)

  @property
  def size(self):
    if self.force_size is not None:
      return self.force_size
    return [self.cnn_net.image_height, self.cnn_net.image_width]


class CNNNSRModelConfig(CNNGeneralModelConfig):

  def __init__(self, **kwargs):
    super(CNNNSRModelConfig, self).__init__(**kwargs)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.data_dir_path = os.path.join(current_dir, '../../data/train')
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
