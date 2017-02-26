
Train a tensorflow model against SVHN dataset, details of which can be found here: http://ufldl.stanford.edu/housenumbers/.

### Download and extract data

```bash
mkdir data
cd data
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz \
    http://ufldl.stanford.edu/housenumbers/test.tar.gz \
    http://ufldl.stanford.edu/housenumbers/extra.tar.gz
tar xzvf train.tar.gz test.tar.gz extra.tar.gz
```


### pre-process

```bash

python3 nsrec/data_preprocessor.py \
    --metadata_file_path=./data/train/digitStruct.mat \
    --output_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data/train

python3 nsrec/data_preprocessor.py \
    --metadata_file_path=./data/test/digitStruct.mat \
    --data_dir_path=./data/test \
    --output_file_path=./data/test/metadata.pickle \
    --rand_bbox_count=0

python3 nsrec/data_preprocessor.py \
    --metadata_file_path=./data/extra/digitStruct.mat \
    --data_dir_path=./data/extra \
    --output_file_path=./data/extra/metadata.pickle \
    --rand_bbox_count=0

python3 nsrec/data_preprocessor.py \
    --metadata_file_path=./data/extra/digitStruct.mat,./data/train/digitStruct.mat \
    --data_dir_path=./data/extra,./data/train \
    --output_file_path=./data/extra-train.raw.tfrecords \
    --rand_bbox_count=0

python3 nsrec/data_preprocessor.py \
    --metadata_file_path=./data/test/digitStruct.mat \
    --data_dir_path=./data/test \
    --output_file_path=./data/test.raw.tfrecords \
    --rand_bbox_count=0

```

### train:

```bash
python3 nsrec/train.py \
    --data_file_path=./data/extra-train.raw.tfrecords \
    --log_every_n_steps=50
```

### infer:

```bash
python3 nsrec/inference.py --net_type=lenet_v2 --input_files=1.png,2.png,3.png,4.png,5.png \
    --checkpoint_dir=./output/train \
    --metadata_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data/train
```

### evaluate against test dataset:

```bash
python3 nsrec/evaluate.py --net_type=lenet_v2 --data_file_path=./data/test.raw.tfrecords
```

### export model to be used in Android application

```bash
python3 nsrec/model_export.py --net_type=lenet_v2
```

### work with bbox

```
python3 nsrec/train.py \
    --data_file_path=./data/extra-train.raw.tfrecords \
    --log_every_n_steps=50 --num_preprocess_threads=30 --number_of_steps=30000

python3 nsrec/inference.py --input_files=train/1.png,train/2.png,train/3.png,train/4.png,train/5.png \
    --checkpoint_dir=./output/train \
    --metadata_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data

python3 nsrec/train.py \
    --data_file_path=./data/extra-train.raw.tfrecords \
    --log_every_n_steps=50 --num_preprocess_threads=30 \
    --number_of_steps=50000 --model_type=bbox --train_dir=./output/train-bbox

python3 nsrec/bbox_inference.py --input_files=train/1.png,train/2.png,train/3.png,train/4.png,train/5.png \
    --checkpoint_dir=./output/train-bbox \
    --net_type=iclr_mnr --model_type=bbox \
    --metadata_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data

python3 nsrec/combined_inference.py \
    --input_files=train/1.png,train/2.png,train/3.png,train/4.png,train/5.png \
    --metadata_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data \
    --bbox_net_type=iclr_mnr \
    --bbox_checkpoint_dir=./output/train-bbox/

python3 nsrec/model_export.py
python3 nsrec/bbox_model_export.py --checkpoint_dir=./output/train-bbox/
```