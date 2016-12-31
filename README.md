
Data from: http://ufldl.stanford.edu/housenumbers/

http://ufldl.stanford.edu/housenumbers/train_32x32.mat
http://ufldl.stanford.edu/housenumbers/test_32x32.mat
http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

### Steps

1. read data
2. BatchGenerator
3. model
4. evaluate

### Preprocessor

preprocess data:

```bash

python3 nsrec/data_preprocessor.py \
    --mat_metadata_file_path=./data/train/digitStruct.mat \
    --output_file_path=./data/train/metadata.pickle \
    --data_dir_path=./data/train

python3 nsrec/data_preprocessor.py \
    --mat_metadata_file_path=./data/test/digitStruct.mat \
    --data_dir_path=./data/test \
    --output_file_path=./data/test/metadata.pickle \
    --rand_bbox_count=0
```

### train:

```bash
python3 nsrec/train.py \
    --metadata_file_path=./data/train/metadata.pickle \
    --log_every_n_steps=50
```
