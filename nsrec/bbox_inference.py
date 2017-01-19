import tensorflow as tf

from inference_base import inference

def main(_):
  inference(lambda label, bbox: bbox, False)

if __name__ == "__main__":
  tf.app.run()
