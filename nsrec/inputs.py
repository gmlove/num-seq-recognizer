import os

from six.moves import cPickle as pickle
import tensorflow as tf
from nsrec import data_preprocessor


def batches(data_generator_fn, max_number_length, batch_size, size, num_preprocess_threads=1):
  filenames, length_labels, numbers_labels = data_generator_fn()

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


def mat_metadata_handler(metadata_file_path, max_number_length, data_dir_path):

  def handler():
    filenames, length_labels, numbers_labels = [], [], []
    metadata = data_preprocessor.metadata_generator(metadata_file_path)

    read_count = 0
    for data in metadata:
      read_count += 1
      if len(data.label) > max_number_length:
        tf.logging.info('ignore data since label is too long: filename=%s, label=%s' % (data.filename, data.label))
        continue

      filename, length_label, numbers_label = _to_data(data.filename, data.label, max_number_length, data_dir_path)
      filenames.append(filename)
      length_labels.append(length_label)
      numbers_labels.append(numbers_label)

      if read_count % 1000 == 0:
        tf.logging.info('readed %s records', read_count)
    return filenames, length_labels, numbers_labels


  return handler


def _to_data(filename, label, max_number_length, data_dir_path):
  numbers_one_hot = [tf.one_hot(ord(ch) - ord('0'), 10) for ch in label]
  no_number_one_hot = tf.constant([[0.1] * 10] * (max_number_length - len(label)))
  filename = os.path.join(data_dir_path, filename)
  length_label = tf.one_hot(len(label) - 1, max_number_length)
  if len(label) < max_number_length:
    numbers_label = tf.concat(0, [numbers_one_hot, no_number_one_hot])
  else:
    numbers_label = numbers_one_hot
  return filename, length_label, numbers_label


def pickle_metadata_handler(metadata_file_path, max_number_length, data_dir_path):

  def handler():
    filenames, length_labels, numbers_labels = [], [], []
    metadata = pickle.load(open(metadata_file_path, 'rb'))
    short_filenames, labels = metadata['filenames'], metadata['labels']

    for i, filename in enumerate(short_filenames):
      filename, length_label, numbers_label = _to_data(filename, labels[i], max_number_length, data_dir_path)
      filenames.append(filename)
      length_labels.append(length_label)
      numbers_labels.append(numbers_label)

    return filenames, length_labels, numbers_labels

  return handler
