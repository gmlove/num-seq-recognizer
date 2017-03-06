import tensorflow as tf

from inference_base import inference
from nsrec import ArgumentsObj

'''
python3 nsrec/train.py \
    --data_file_path=./data/extra-train.number0.raw.tfrecords \
    --log_every_n_steps=50 --num_preprocess_threads=30 \
    --number_of_steps=50000 --model_type=bbox \
    --train_dir=./output/train-bbox-l2 --learning_rate=0.00001 --optimizer=RMSProp
'''
def main(_):
  args = ArgumentsObj().define_arg('model_type', 'bbox')
  inference(lambda label, bbox: bbox, flags=args)

if __name__ == "__main__":
  tf.app.run()
