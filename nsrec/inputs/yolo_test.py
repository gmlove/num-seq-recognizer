import numpy as np
import tensorflow as tf
from nsrec import test_helper
from nsrec.inputs import yolo


class TestYOLO(tf.test.TestCase):

  def test_batches(self):
    data_file_path = test_helper.get_test_metadata()
    batch_size, size, B, H, W, C = 2, (416, 416), 5, 13, 13, 10
    first_loss_batch = self._calculate_loss_feed_batches(0)
    second_loss_batch = self._calculate_loss_feed_batches(1)
    third_loss_batch = self._calculate_loss_feed_batches(2)
    with self.test_session() as sess:
      data_batches, loss_feed_batches, _, _ = \
        yolo.batches(data_file_path, 5, batch_size, size, num_preprocess_threads=1, channels=3, is_training=False)

      self.assertEqual(data_batches.get_shape(), (2, 416, 416, 3))
      self.assertEqual(loss_feed_batches['probs'].get_shape(), (2, H * W, B, C))
      self.assertEqual(loss_feed_batches['confs'].get_shape(), (2, H * W, B))
      self.assertEqual(loss_feed_batches['coord'].get_shape(), (2, H * W, B, 4))
      self.assertEqual(loss_feed_batches['proid'].get_shape(), (2, H * W, B, C))
      self.assertEqual(loss_feed_batches['areas'].get_shape().as_list(), [2, H * W, B])
      self.assertEqual(loss_feed_batches['upleft'].get_shape(), (2, H * W, B, 2))
      self.assertEqual(loss_feed_batches['botright'].get_shape(), (2, H * W, B, 2))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      sess.run(tf.local_variables_initializer())
      _, lfb = sess.run([data_batches, loss_feed_batches])

      all_keys = ['probs', 'confs', 'coord', 'proid', 'areas', 'upleft', 'botright']

      for k in all_keys:
        print(second_loss_batch)
        self.assertAllClose(lfb[k][0], first_loss_batch[k])
        self.assertAllClose(lfb[k][1], second_loss_batch[k])

      _, lfb = sess.run([data_batches, loss_feed_batches])
      for k in all_keys:
        self.assertAllClose(lfb[k][0], third_loss_batch[k])

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def test_print_first5_test_data(self):
    self.test_print_test_data(list(range(5)))

  def test_print_test_data(self, indices=None):
    indices = indices or [0]
    with self.test_session() as sess:
      filename_queue = tf.train.string_input_producer([test_helper.test_data_file])
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
        serialized_example,
        features={
          'image_png': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([5], tf.int64),
          'length': tf.FixedLenFeature([1], tf.int64),
          'bbox': tf.FixedLenFeature([4], tf.int64),
          'sep_bbox_list': tf.FixedLenFeature([4 * 5], tf.int64)
        })
      image = tf.image.decode_png(features['image_png'], 3)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      for i in indices:
        image_, test_data = sess.run([image, features])
        print('image[%s] shape: ' % i, image_.shape)
        del test_data['image_png']
        print('\n'.join(['%s: %s' % (k, v) for k, v in dict(test_data).items()]))

      coord.request_stop()
      coord.join(threads)
      sess.close()

  def test_preprocess_image(self):
    image = tf.zeros([200, 200, 3], dtype=tf.uint8)
    bboxes = tf.constant([[120, 120, 30, 30], [150, 150, 40, 40], [0, 0, 0, 0]])
    main_bbox = tf.constant([120, 120, 70, 70])
    with self.test_session() as sess:
      from nsrec.inputs.yolo import preprocess_image
      print(sess.run(preprocess_image(image, main_bbox, bboxes, [400, 400], True)))

  def _test_data(self, index):
    wh = [[741, 350], [199, 83], [52, 23]]
    allobj = [[['1', 246, 77, 246 + 81, 77 + 219],
               ['9', 323, 81, 323 + 96, 81 + 219]],
              [['2', 77, 29, 77 + 23, 29 + 32],
               ['3', 98, 25, 98 + 26, 25 + 32]],
              [['2', 17, 5, 17 + 8, 5 + 15],
               ['5', 25, 5, 25 + 9, 5 + 15]]]
    return wh[index][0], wh[index][1], allobj[index]

  def _calculate_loss_feed_batches(self, index):
    B, C, H, W = 5, 10, 13, 13
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    w, h, allobj = self._test_data(index)

    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
      centerx = .5*(obj[1]+obj[3]) #xmin, xmax
      centery = .5*(obj[2]+obj[4]) #ymin, ymax
      cx = centerx / cellx
      cy = centery / celly
      if cx >= W or cy >= H:
        print('image error')
        return None, None
      obj[3] = float(obj[3]-obj[1]) / w
      obj[4] = float(obj[4]-obj[2]) / h
      obj[3] = np.sqrt(obj[3])
      obj[4] = np.sqrt(obj[4])
      obj[1] = cx - np.floor(cx) # centerx
      obj[2] = cy - np.floor(cy) # centery
      obj += [int(np.floor(cy) * W + np.floor(cx))]

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    for obj in allobj:
      probs[obj[5], :, :] = [[0.]*C] * B
      probs[obj[5], :, labels.index(obj[0])] = 1.
      proid[obj[5], :, :] = [[1.]*C] * B
      coord[obj[5], :, :] = [obj[1:5]] * B
      prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * W # xleft
      prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * H # yup
      prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * W # xright
      prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * H # ybot
      confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft;
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at loss layer
    loss_feed_val = {
      'probs': probs, 'confs': confs,
      'coord': coord, 'proid': proid,
      'areas': areas, 'upleft': upleft,
      'botright': botright
    }

    print(['%s: %s' % (k, v.shape) for k, v in loss_feed_val.items()])

    return loss_feed_val

