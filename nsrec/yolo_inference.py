import tensorflow as tf

from inference_base import inference
from nsrec import ArgumentsObj

'''
python3 nsrec/inference.py --checkpoint_dir=./output/train-yolo/ \
  --input_files=train/1.png,train/2.png,train/3.png,train/4.png,train/5.png \
  --metadata_file_path=./data/train/metadata.pickle \
  --data_dir_path=./data --threshold=0.1
'''
def main(_):
  args = ArgumentsObj()\
    .define_arg('model_type', 'yolo')\
    .define_arg('net_type', 'yolo')
  inference(lambda label, bbox: label, bboxes=None, flags=args)

if __name__ == "__main__":
  tf.app.run()
