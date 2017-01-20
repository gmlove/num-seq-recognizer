import tensorflow as tf

from inference_base import combined_inference

tf.flags.DEFINE_string("bbox_net_type", "lenet", "Which net to use: lenet or alexnet")
tf.flags.DEFINE_string("bbox_checkpoint_dir", None,
                       "Directory containing model checkpoints.")


def main(_):
  combined_inference()


if __name__ == "__main__":
  tf.app.run()
