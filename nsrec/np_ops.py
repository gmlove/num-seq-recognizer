import numpy as np

def one_hot(num, max_num):
  return np.eye(max_num)[num - 1]


