import numpy as np


def one_hot(num, max_num):
  assert 0 < num <= max_num, '0 < num <= max_num, max_num=%s, num=%s' % (max_num, num)
  return np.eye(max_num)[num - 1]


def correct_count(length_label_batches, numbers_label_batches,
                  length_label_batches_pd, numbers_label_batches_pd):
  batch_size, max_number_length = len(length_label_batches), len(numbers_label_batches[0])
  count = 0
  for i in range(batch_size):
    length = np.argmax(length_label_batches[i]) + 1
    length_correct = np.argmax(length_label_batches_pd[i]) == length - 1
    all_correct = [length_correct]
    for j in range(length):
      number = np.argmax(numbers_label_batches[j][i])
      number_predict = np.argmax(numbers_label_batches_pd[j][i])
      all_correct.append(number == number_predict)
    if np.alltrue(all_correct):
      count += 1
  return count
