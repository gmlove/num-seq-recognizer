import os

import tensorflow as tf
from nsrec import data_reader

def batches(metadata_file_path, data_dir_path, max_number_length, batch_size, size, num_preprocess_threads=1):
  def input_data_generator():
    filenames, length_labels, numbers_labels = [], [], []
    metadata = data_reader.metadata_generator(metadata_file_path)
    for data in metadata:
      if len(data.label) > max_number_length:
        tf.logging.info('ignore data since label is too long: filename=%s, label=%s' % (data.filename, data.label))
        continue
      numbers_one_hot = [tf.one_hot(ord(ch) - ord('0'), 10) for ch in data.label]
      no_number_one_hot = tf.constant([[0.1] * 10] * (max_number_length - len(data.label)))
      filenames.append(os.path.join(data_dir_path, data.filename))
      length_labels.append(tf.one_hot(len(data.label) - 1, max_number_length))
      if len(data.label) < max_number_length:
        numbers_labels.append(tf.concat(0, [numbers_one_hot, no_number_one_hot]))
      else:
        numbers_labels.append(numbers_one_hot)
    return filenames, length_labels, numbers_labels

  filenames, length_labels, numbers_labels = input_data_generator()

  filename_queue = tf.train.string_input_producer(
    filenames, shuffle=False, capacity=batch_size * 3)
  length_label_queue = tf.train.input_producer(
    length_labels, shuffle=False, element_shape=(max_number_length, ), capacity=batch_size * 3)
  numbers_label_queue = tf.train.input_producer(
    numbers_labels, shuffle=False, element_shape=(max_number_length, 10), capacity=batch_size * 3)

  reader = tf.WholeFileReader()
  _, dequeued_record_string = reader.read(filename_queue)
  dequeued_img = tf.image.decode_png(dequeued_record_string, 3)
  dequeued_img = tf.image.resize_images(dequeued_img, size)

  return tf.train.batch(
    [dequeued_img, length_label_queue.dequeue(), numbers_label_queue.dequeue()],
    batch_size=batch_size, capacity=batch_size * 3)
