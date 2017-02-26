import tensorflow as tf

from inference_base import inference
from nsrec import ArgumentsObj


def main(_):
  args = ArgumentsObj().define_arg('model_type', 'bbox')
  inference(lambda label, bbox: bbox, flags=args)

if __name__ == "__main__":
  tf.app.run()
