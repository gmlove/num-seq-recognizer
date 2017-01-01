import tensorflow as tf

tensors_to_inspect = {}

def inspect_tensors(sess, feed_dict=None):
  inspected = sess.run(tensors_to_inspect, feed_dict)
  for k, v in inspected:
    tf.logging.debug('inspected tensor: key=%s, value=%s', k, v)
