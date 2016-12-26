
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
python nsrec/data_preprocessor.py \
    --metadata_file_path=./data/train/digitStruct.mat \
    --output_file_path=./data/train/metadata.pickle
```

### train:

```bash
python3 nsrec/train.py \
    --metadata_file_path=./data/train/metadata.pickle \
    --log_every_n_steps=50
```
