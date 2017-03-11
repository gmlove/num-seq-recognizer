import numpy as np
import tensorflow as tf


def _preprocess_image(image, bboxes, size):
  # TODO: transform image with random noise
  image = tf.image.resize_images(image, size)
  return image, bboxes


def batches(data_file_path, max_number_length, batch_size, size,
            num_preprocess_threads=1, is_training=True, channels=3):
  filename_queue = tf.train.string_input_producer([data_file_path])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image_png': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([max_number_length], tf.int64),
      'length': tf.FixedLenFeature([1], tf.int64),
      'bbox': tf.FixedLenFeature([4], tf.int64),
      'sep_bbox_list': tf.FixedLenFeature([4 * max_number_length], tf.int64)
    })
  image, sep_bbox_list, label, length = features['image_png'], features['sep_bbox_list'], features['label'], features['length']
  bboxes, label = tf.cast(sep_bbox_list, tf.float32), tf.cast(label, tf.int32)
  bboxes = tf.reshape(bboxes, [max_number_length, 4])

  image = tf.image.decode_png(image, channels)
  image_shape = tf.shape(image)
  image, bboxes = _preprocess_image(image, bboxes, size)

  dequeued_data = []
  for i in range(num_preprocess_threads):
    dequeued_data.append([image, image_shape, bboxes, label])
  image_batch, image_shape_batch, bboxes_batch, label_batch = \
    tf.train.batch_join(dequeued_data, batch_size=batch_size, capacity=batch_size * 3)
  label_bboxes_batch = bboxes_batch

  H, W, C, B = 13, 13, 10, max_number_length
  h, w = image_shape_batch[:, 0], image_shape_batch[:, 1]

  cellx, celly = tf.cast(w, tf.float32) / W, tf.cast(h, tf.float32) / H
  centerx = bboxes_batch[:, :, 0] + 0.5 * bboxes_batch[:, :, 2]
  centery = bboxes_batch[:, :, 1] + 0.5 * bboxes_batch[:, :, 3]
  cx, cy = centerx / tf.expand_dims(cellx, 1), centery / tf.expand_dims(celly, 1)
  w = tf.expand_dims(tf.cast(w, tf.float32), 1)
  h = tf.expand_dims(tf.cast(h, tf.float32), 1)
  bboxes_batch = tf.concat([
    tf.expand_dims(cx - tf.floor(cx), 2),
    tf.expand_dims(cy - tf.floor(cy), 2),
    tf.expand_dims(tf.sqrt(bboxes_batch[:, :, 2] / w), 2),
    tf.expand_dims(tf.sqrt(bboxes_batch[:, :, 3] / h), 2)
  ], axis=2)

  indices = tf.cast(tf.floor(cy) * W + tf.floor(cx), tf.int32)
  probs = tf.contrib.framework.local_variable(np.zeros([batch_size, W * H, B, C], dtype=np.float32))
  confs = tf.contrib.framework.local_variable(np.zeros([batch_size, W * H, B], dtype=np.float32))
  coord = tf.contrib.framework.local_variable(np.zeros([batch_size, W * H, B, 4], dtype=np.float32))
  proid = tf.contrib.framework.local_variable(np.zeros([batch_size, W * H, B, C], dtype=np.float32))
  prear = tf.contrib.framework.local_variable(np.zeros([batch_size, W * H, 4], dtype=np.float32))

  probs_init = probs.assign(tf.zeros([batch_size, W * H, B, C], dtype=tf.float32))
  confs_init = confs.assign(tf.zeros([batch_size, W * H, B], dtype=tf.float32))
  coord_init = coord.assign(tf.zeros([batch_size, W * H, B, 4], dtype=tf.float32))
  proid_init = proid.assign(tf.zeros([batch_size, W * H, B, C], dtype=tf.float32))
  prear_init = prear.assign(tf.zeros([batch_size, W * H, 4], dtype=tf.float32))

  with tf.control_dependencies([probs_init, confs_init, coord_init, proid_init, prear_init]):
    assign_ops = []
    for i in range(batch_size):
      indices_i = indices[i] # [5]
      labels_i = label_batch[i]
      bboxes_i = bboxes_batch[i]
      for j in range(B):
        to_assign = tf.one_hot(labels_i[j], C, dtype=tf.float32)
        probs_i_j = probs[i, indices_i[j]].assign([to_assign] * B)
        proid_i_j = tf.cond(bboxes_i[j, 3] < 1e-6, lambda: proid, lambda: proid[i, indices_i[j]].assign([[1.] * C] * B))
        coord_i_j = coord[i, indices_i[j]].assign([bboxes_i[j]] * B)
        prear_i_j = prear[i, indices_i[j]].assign([
          bboxes_i[j, 0] - bboxes_i[j, 2] ** 2 * .5 * W,
          bboxes_i[j, 1] - bboxes_i[j, 3] ** 2 * .5 * H,
          bboxes_i[j, 0] + bboxes_i[j, 2] ** 2 * .5 * W,
          bboxes_i[j, 1] + bboxes_i[j, 3] ** 2 * .5 * H
        ])
        confs_i_j = tf.cond(bboxes_i[j, 3] < 1e-6, lambda: confs, lambda: confs[i, indices_i[j]].assign([1] * B))
        assign_ops.extend([probs_i_j, proid_i_j, coord_i_j, prear_i_j, confs_i_j])

  with tf.control_dependencies(assign_ops):
    upleft = tf.expand_dims(prear[:, :, 0:2], 2)
    botright = tf.expand_dims(prear[:, :, 2:4], 2)
    wh = botright - upleft
    area = wh[:, :, :, 0] * wh[:, :, :, 1]
    upleft = tf.concat([upleft] * B, 2)
    botright = tf.concat([botright] * B, 2)
    areas = tf.concat([area] * B, 2)

  loss_feed = {
    'probs': probs, 'confs': confs,
    'coord': coord, 'proid': proid,
    'areas': areas, 'upleft': upleft,
    'botright': botright
  }

  return image_batch, loss_feed, label_batch, label_bboxes_batch
