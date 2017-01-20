import tensorflow as tf

from inference_base import inference

def main(_):
  inference(lambda label, bbox: bbox, None)

if __name__ == "__main__":
  tf.app.run()
