import tensorflow as tf


class Inferrable(object):

  def __init__(self, graph_file_path, initializer_node_name, input_node_name, output_node_name):
    self.graph = tf.Graph()
    self.session = tf.Session(graph=self.graph)

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(open(graph_file_path, 'rb').read())
    with self.graph.as_default():
      tf.import_graph_def(graph_def)

    if initializer_node_name:
      self.initializer = self.graph.get_operation_by_name('import/' + initializer_node_name)
    self.input = self.graph.get_tensor_by_name('import/%s:0' % input_node_name)
    self.output = self.graph.get_tensor_by_name('import/%s:0' % output_node_name)

    if initializer_node_name:
      self.session.run(self.initializer)

  def infer(self, input_data):
    return self.session.run(self.output, feed_dict={self.input: input_data})


