import tensorflow as tf

from inference_base import inference
from nsrec import ArgumentsObj

tf.flags.DEFINE_string("bbox_net_type", "lenet", "Which net to use: lenet or alexnet")
tf.flags.DEFINE_string("bbox_checkpoint_dir", None,
                       "Directory containing model checkpoints.")


def main(_):
  args = ArgumentsObj('bbox').define_arg('model_type', 'bbox')
  bboxes = inference(lambda labels, bboxes: bboxes, None, args)
  if not bboxes:
    raise Exception("Bbox not calculated.")
  inference(lambda labels, bboxes: labels, bboxes)


if __name__ == "__main__":
  tf.app.run()
