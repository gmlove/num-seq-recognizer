import math

import tensorflow as tf
import numpy as np

from nsrec.inputs import inputs
from nsrec.inputs import yolo as yolo_inputs
from tensorflow.python.framework import ops
from nsrec.utils.ops import global_step_variable, gray_scale, assign_vars, all_model_variables_data


def expit_tensor(x):
  return 1. / (1. + tf.exp(-x))


def expit(x):
  return 1. / (1. + np.exp(-x))


def _softmax(x):
  e_x = np.exp(x - np.max(x))
  out = e_x / e_x.sum()
  return out


def prob_compare(box):
  return box.probs[box.class_num]


def overlap(x1, w1, x2, w2):
  l1 = x1 - w1 / 2.
  l2 = x2 - w2 / 2.
  left = max(l1, l2)
  r1 = x1 + w1 / 2.
  r2 = x2 + w2 / 2.
  right = min(r1, r2)
  return right - left


def box_intersection(a, b):
  w = overlap(a.x, a.w, b.x, b.w)
  h = overlap(a.y, a.h, b.y, b.h)
  if w < 0 or h < 0: return 0
  area = w * h
  return area


def box_union(a, b):
  i = box_intersection(a, b)
  u = a.w * a.h + b.w * b.h - i
  return u


def box_iou(a, b):
  return box_intersection(a, b) / box_union(a, b)


def extract_label(net_out, max_number_length, num_classes, threshold):
  B, C, threshold = max_number_length, num_classes, threshold
  lefts, ltwh_bboxes, max_indices, _ = extract_label_as_array(net_out, H, W, B, C, threshold)

  left_labels = []
  for i in range(len(lefts)):
    left_labels.append({'left': lefts[i], 'label': max_indices[i], 'bbox': ltwh_bboxes[i]})

  return left_labels


def extract_label_as_array(net_out, H, W, B, C, threshold):
  net_out = net_out.reshape([H, W, B, -1])
  boxes = list()
  for row in range(H):
    for col in range(W):
      for b in range(B):
        bx = BoundBox(C)
        bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
        bx.c = expit(bx.c)
        bx.x = (col + expit(bx.x)) / W
        bx.y = (row + expit(bx.y)) / H
        bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
        bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
        classes = net_out[row, col, b, 5:]
        bx.probs = _softmax(classes) * bx.c
        bx.probs *= bx.probs > threshold
        boxes.append(bx)

  # non max suppress boxes
  for c in range(C):
    for i in range(len(boxes)):
      boxes[i].class_num = c
    boxes = sorted(boxes, key=prob_compare, reverse=True)
    for i in range(len(boxes)):
      boxi = boxes[i]
      if boxi.probs[c] == 0: continue
      for j in range(i + 1, len(boxes)):
        boxj = boxes[j]
        if box_iou(boxi, boxj) >= .4:
          boxes[j].probs[c] = 0.
  lefts, ltwh_bboxes, selected_classes, selected_classes_probs = [], [], [], []
  for b in boxes:
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    if max_prob > threshold:
      selected_classes.append(max_indx)
      selected_classes_probs.append(max_prob)
      left = (b.x - b.w / 2.)
      left = 0 if left < 0 else left
      lefts.append(left)
      ltwh_bbox = [left, b.y - b.h / 2, b.w, b.h]
      ltwh_bboxes.append(ltwh_bbox)
  indices = np.argsort(np.array(lefts))
  lefts, sorted_classes, sorted_classes_probs, ltwh_bboxes = \
    [np.array(data)[indices] for data in [lefts, selected_classes, selected_classes_probs, ltwh_bboxes]]
  return lefts, ltwh_bboxes, sorted_classes, sorted_classes_probs


def align_label(label, max_number_length):
  label = label[:max_number_length]
  if len(label) < max_number_length:
    label += [10] * (max_number_length - len(label))
  return label


H, W, C = 13, 13, 10
HW = H * W # number of grid cells
anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]


class BoundBox:

  def __init__(self, classes):
    self.x, self.y = float(), float()
    self.w, self.h = float(), float()
    self.c = float()
    self.class_num = classes
    self.probs = np.zeros((classes,))


class YOLOTrainModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.is_training = True
    self.max_number_length = config.max_number_length
    self.batch_size = config.batch_size

    self.total_loss = None
    self.global_step = None
    self.net_out = None
    self.fetch = None

    self.data_batches = None
    self.loss_feed_batches = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, origin_image_shape_batch, image_shape_batch, label_batch, label_bboxes_batch = \
        yolo_inputs.batches(config.data_file_path, self.max_number_length, config.batch_size, config.size,
                     num_preprocess_threads=config.num_preprocess_threads, channels=config.channels)
      self.loss_feed_batches = yolo_inputs.prepare_for_loss(
        self.max_number_length, config.batch_size, label_bboxes_batch, image_shape_batch, label_batch)

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as vs:
      collection_name = self.cnn_net.end_points_collection_name(vs)
      self.net_out, _ = self.cnn_net.cnn_layers(
        self.data_batches, vs, collection_name, is_training=self.is_training)

  def _setup_loss(self):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    number_losses = []
    net_out = self.net_out
    with ops.name_scope(None, 'Loss'):
      sconf = object_scale = 5
      snoob = noobject_scale = 1
      sprob = class_scale = 1
      scoor = coord_scale = 1

      B = self.max_number_length

      size1 = [None, HW, B, C]
      size2 = [None, HW, B]

      # return the below placeholders
      _probs = self.loss_feed_batches['probs']
      _confs = self.loss_feed_batches['confs']
      _coord = self.loss_feed_batches['coord']
      # weights term for L2 loss
      _proid = self.loss_feed_batches['proid']
      # material calculating IOU
      _areas = self.loss_feed_batches['areas']
      _upleft = self.loss_feed_batches['upleft']
      _botright = self.loss_feed_batches['botright']

      # Extract the coordinate prediction from net.out
      net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, H*W, B, 4])
      adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
      area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
      centers = coords[:, :, :, 0:2]
      floor = centers - (wh * .5)
      ceil = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU, set 0.0 confidence for worse boxes
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
      best_box = tf.to_float(best_box)
      confs = tf.multiply(best_box, _confs)

      # take care of the weight terms
      conid = snoob * (1. - confs) + sconf * confs
      weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
      cooid = scoor * weight_coo
      weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
      proid = sprob * weight_pro

      self.fetch = [_probs, confs, conid, cooid, proid]
      true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
      wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

      loss = tf.pow(adjusted_net_out - true, 2)
      loss = tf.multiply(loss, wght)
      loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
      loss = tf.reduce_sum(loss, 1)
      self.total_loss = .5 * tf.reduce_mean(loss)

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


class YOLOEvalModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.is_training = False
    self.max_number_length = config.max_number_length
    self.batch_size = config.batch_size

    self.net_out = None
    self.fetch = None

    self.data_batches = None
    self.label_batches = None
    self.label_bboxes_batches = None
    self.image_shape = None

  def _setup_input(self):
    config = self.config
    with ops.name_scope(None, 'Input') as sc:
      self.data_batches, self.image_shape, _, self.label_batches, self.label_bboxes_batches = \
        yolo_inputs.batches(config.data_file_path, self.max_number_length, config.batch_size, config.size,
                     num_preprocess_threads=config.num_preprocess_threads, channels=config.channels)

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as vs:
      collection_name = self.cnn_net.end_points_collection_name(vs)
      self.net_out, _ = self.cnn_net.cnn_layers(
        self.data_batches, vs, collection_name, is_training=self.is_training)

  def _setup_global_step(self):
    self.global_step = global_step_variable()

  def build(self):
    self._setup_input()
    self._setup_net()
    self._setup_global_step()

  def correct_count(self, sess):
    net_out, image_shape, label_batches, _ = sess.run(
      [self.net_out, self.image_shape, self.label_batches, self.label_bboxes_batches])
    labels = []
    for net_out_i in net_out:
      label = extract_label(net_out_i, self.max_number_length, self.config.num_classes, self.config.threshold)
      labels.append(align_label([l['label'] for l in label], self.max_number_length))
    return sum([1 if all(label_batches[i] == labels[i]) else 0 for i in range(self.batch_size)])


class YOLOInferModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.max_number_length = config.max_number_length
    self.is_training = False

    self.inputs = None
    self.data_batches = None
    self.image_shape = None

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self):
    with self.cnn_net.variable_scope([self.data_batches]) as vs:
      collection_name = self.cnn_net.end_points_collection_name(vs)
      self.net_out, _ = self.cnn_net.cnn_layers(
        self.data_batches, vs, collection_name, is_training=self.is_training)

  def build(self):
    self._setup_input()
    self._setup_net()

  def infer(self, sess, data):

    def to_coordinate_bboxes(label, image_shape):
      def to_coordinate_bbox(bbox):
        w, h = image_shape[1], image_shape[0]
        return list(map(int, [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]))
      return [to_coordinate_bbox(l['bbox']) for l in label]

    input_data = [inputs.normalize_img(image, self.config.size) for image in data]
    net_out = sess.run(self.net_out, feed_dict={self.inputs: input_data})
    labels = []
    for net_out_i in net_out:
      labels.append(extract_label(net_out_i, self.max_number_length, self.config.num_classes, self.config.threshold))
    join_label = lambda label: ''.join(map(lambda l: str(l['label']), label))
    return [(join_label(labels[i]), to_coordinate_bboxes(labels[i], data[i].shape))
            for i in range(len(labels))]

  def vars(self, sess):
    return all_model_variables_data(sess)


def build_export_output(net_out, H, W, max_number_length, C, threshold):
  B, threshold, sprs_output_box_count = max_number_length, threshold, 100
  net_out = tf.reshape(net_out, [H, W, B, -1])
  boxes, boxes_scores, classes_probs = net_out[:, :, :, :4], net_out[:, :, :, 4], net_out[:, :, :, 5:]
  row = np.concatenate([np.ones([1, W, B], dtype=np.float32) * i for i in range(H)], axis=0)
  col = np.concatenate([np.ones([H, 1, B], dtype=np.float32) * i for i in range(W)], axis=1)
  anchors_w = np.concatenate([np.ones([H, W, 1], dtype=np.float32) * anchors[2 * i + 0] for i in range(B)], axis=2)
  anchors_h = np.concatenate([np.ones([H, W, 1], dtype=np.float32) * anchors[2 * i + 1] for i in range(B)], axis=2)
  with ops.name_scope(None, 'calc_boxes_coordinates'):
    boxes = tf.concat([
      tf.expand_dims((tf.sigmoid(boxes[:, :, :, 0]) + col) / W, 3),
      tf.expand_dims((tf.sigmoid(boxes[:, :, :, 1]) + row) / H, 3),
      tf.expand_dims(tf.exp(boxes[:, :, :, 2]) * anchors_w / W, 3),
      tf.expand_dims(tf.exp(boxes[:, :, :, 3]) * anchors_h / H, 3),
    ], axis=3)
    boxes = tf.cast(boxes, tf.float32)

  with ops.name_scope(None, 'calc_boxes_scores'):
    boxes_scores = tf.sigmoid(boxes_scores)
    boxes_scores = tf.nn.softmax(classes_probs) * tf.expand_dims(boxes_scores, 3)
    boxes_scores = boxes_scores * tf.cast(boxes_scores > threshold, tf.float32)
    boxes_scores = tf.cast(boxes_scores, tf.float32)

  with ops.name_scope(None, 'non_max_suppression'):
    boxes = tf.reshape(boxes, [H * W * B, 4])
    sprs_boxes, sprs_boxes_scores = [], []
    for i in range(C):
      box_scores = tf.reshape(boxes_scores[:, :, :, i], [H * W * B])
      sprs_boxes_indices = tf.image.non_max_suppression(boxes, box_scores, sprs_output_box_count, iou_threshold=0.4)
      box_scores = box_scores * tf.scatter_nd(
        tf.reshape(sprs_boxes_indices, [-1, 1]),
        tf.ones(tf.shape(sprs_boxes_indices), dtype=tf.float32), [H * W * B])
      sprs_boxes_scores.append(tf.reshape(box_scores, [H * W * B, 1]))

  with ops.name_scope(None, 'select_boxes'):
    sprs_boxes_scores = tf.concat(sprs_boxes_scores, axis=1)
    classes = tf.argmax(sprs_boxes_scores, axis=1)
    classes_probs = tf.reduce_max(sprs_boxes_scores, axis=1)

    selected_box_mask = classes_probs > threshold

    selected_classes = tf.boolean_mask(classes, selected_box_mask)
    selected_boxes = tf.boolean_mask(boxes, selected_box_mask)
    selected_classes_probs = tf.boolean_mask(classes_probs, selected_box_mask)
    lefts = selected_boxes[:, 0] - selected_boxes[:, 2] / 2
    lefts = tf.where(lefts < 0, tf.zeros(tf.shape(lefts)), lefts)
    selected_boxes = tf.concat([
      tf.expand_dims(lefts, 1),
      tf.expand_dims(selected_boxes[:, 1] - selected_boxes[:, 3] / 2, 1),
      tf.expand_dims(selected_boxes[:, 2], 1),
      tf.expand_dims(selected_boxes[:, 3], 1),
    ], axis=1)
    selected_lefts = selected_boxes[:, 0]

  with ops.name_scope(None, 'sort_boxes'):
    sorted_lefts, sorted_lefts_indices = tf.nn.top_k(selected_lefts * -1, tf.shape(selected_lefts)[0])

    sorted_classes = tf.gather(selected_classes, sorted_lefts_indices)
    sorted_boxes = tf.gather(selected_boxes, sorted_lefts_indices)
    sorted_classes_probs = tf.gather(selected_classes_probs, sorted_lefts_indices)
  return sorted_lefts * -1, sorted_boxes, sorted_classes, sorted_classes_probs


class YOLOToExportModel:

  def __init__(self, config):
    self.config = config
    self.cnn_net = config.cnn_net
    self.max_number_length = config.max_number_length
    self.is_training = False

    self.inputs = None
    self.data_batches = None

    self.initializer = None
    self.output = None
    self.output_boxes = None
    self.output_classes = None
    self.output_classes_probs = None

  def _vars(self):
    coll = tf.get_collection(ops.GraphKeys.MODEL_VARIABLES)
    return dict(zip([v.name for v in coll], coll))

  def _setup_input(self):
    self.inputs = tf.placeholder(tf.float32, (None, self.config.size[0], self.config.size[1], 3), name='input')
    self.data_batches = gray_scale(self.inputs) if self.config.gray_scale else self.inputs

  def _setup_net(self, saved_vars_dict):
    with self.cnn_net.variable_scope([self.data_batches]) as vs:
      collection_name = self.cnn_net.end_points_collection_name(vs)
      net_out, _ = self.cnn_net.cnn_layers(
        self.data_batches, vs, collection_name, is_training=self.is_training)
      self.output = tf.identity(net_out, name='output')

      _, self.output_boxes, self.output_classes, self.output_classes_probs = \
          build_export_output(net_out, H, W, self.max_number_length, self.config.num_classes, self.config.threshold)
      self.output_boxes = tf.identity(self.output_boxes, 'output_boxes')
      self.output_classes = tf.identity(self.output_classes, 'output_classes')
      self.output_classes_probs = tf.identity(self.output_classes_probs, 'output_classes_probs')

      assign_ops = assign_vars(self._vars(), saved_vars_dict)
      self.initializer = tf.group(*assign_ops, name='initializer')

  def build(self, saved_vars_dict):
    self._setup_input()
    self._setup_net(saved_vars_dict)
