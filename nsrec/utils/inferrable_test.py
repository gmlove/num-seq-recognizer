import os

import numpy as np
import tensorflow as tf
from nsrec.inputs import inputs
from nsrec.models import CNNBBoxInferModel, CNNBBoxToExportModel, CNNNSRInferModelConfig

from nsrec import test_helper
from nsrec.utils.inferrable import Inferrable


class InferrableTest(tf.test.TestCase):

  def test_infer(self):
    from nsrec.nets import lenet_v2
    self._create_test_graph()
    inferrable = Inferrable(test_helper.test_graph_file,
                            CNNBBoxToExportModel.INITIALIZER_NODE_NAME,
                            CNNBBoxToExportModel.INPUT_NODE_NAME, CNNBBoxToExportModel.OUTPUT_NODE_NAME)
    input_data = inputs.read_img(os.path.join(test_helper.train_data_dir_path, '1.png'))
    input_data = inputs.normalize_img(input_data, [lenet_v2.image_width, lenet_v2.image_height])
    pbs = inferrable.infer(np.array([input_data]))
    print(pbs)

  def _create_test_graph(self):
    config = CNNNSRInferModelConfig(net_type='lenet_v2')

    with self.test_session(tf.Graph()) as sess:
      model = CNNBBoxInferModel(config)
      model.build()
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      model_vars = model.vars(sess)

    with self.test_session(tf.Graph()) as sess:
      model = CNNBBoxToExportModel(config)
      model.build(model_vars)
      sess.run(model.initializer)
      print(tf.train.write_graph(sess.graph_def,
                                 os.path.dirname(test_helper.test_graph_file),
                                 os.path.basename(test_helper.test_graph_file), as_text=False))

  def test_combined_infer(self):
    from nsrec.nets import iclr_mnr, lenet_v2
    from six.moves import cPickle as pickle

    metadata = pickle.loads(open(test_helper.train_data_dir_path + '/metadata.pickle', 'rb').read())

    def test_img_data_generator(new_size, crop_bbox=False):
      for i in range(10):
        filename = '%s.png' % (i + 1)
        img_idx = metadata['filenames'].index(filename)
        bbox, label = metadata['bboxes'][img_idx], metadata['labels'][img_idx]
        input_data = inputs.read_img(os.path.join(test_helper.train_data_dir_path, filename))
        width, height = input_data.shape[1], input_data.shape[0]
        if crop_bbox:
          input_data = inputs.read_img(os.path.join(test_helper.train_data_dir_path, filename), bbox)
        input_data = inputs.normalize_img(input_data, [new_size[0], new_size[1]])
        yield (input_data, (width, height), bbox, label)

    bbox_model = Inferrable(test_helper.output_bbox_graph_file, 'initializer-bbox', 'input-bbox', 'output-bbox')
    for input_data, (width, height), bbox, _ in test_img_data_generator([lenet_v2.image_width, lenet_v2.image_height]):
      bbox_in_rate = bbox_model.infer(np.array([input_data]))
      print(width, height)
      print('label bbox: %s, bbox: %s' % (bbox, [bbox_in_rate[0] * width, bbox_in_rate[1] * height,
                                                 bbox_in_rate[2] * width, bbox_in_rate[3] * height]))

    nsr_model = Inferrable(test_helper.output_graph_file, 'initializer', 'input', 'output')
    for input_data, _, _, label in test_img_data_generator([iclr_mnr.image_width, iclr_mnr.image_height], True):
      pb = nsr_model.infer(np.array([input_data]))
      print('actual: %s, length pb: %s, numbers: %s' % (
        label, np.argmax(pb[:5]), np.argmax(pb[5:].reshape([5, 11]), axis=1)))


if __name__ == '__main__':
  tf.test.main()
