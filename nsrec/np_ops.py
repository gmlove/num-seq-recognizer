import numpy as np


def one_hot(num, max_num):
  return np.eye(max_num)[num - 1]


def correct_count(length_label_batches, numbers_label_batches,
                  length_label_batches_pd, numbers_label_batches_pd):
  batch_size, max_number_length = len(length_label_batches), len(numbers_label_batches[0])
  count = 0
  for i in range(batch_size):
    length = np.argmax(length_label_batches[i]) + 1
    length_correct = np.argmax(length_label_batches_pd[i]) == length - 1
    numbers = np.argmax(numbers_label_batches[i], axis=1)
    numbers_predict = []
    for j in range(max_number_length):
      numbers_predict.append(np.argmax(numbers_label_batches_pd[i][j]))
    if np.alltrue([length_correct] + list((numbers == numbers_predict))[:length]):
      count += 1
  return count
