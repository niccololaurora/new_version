import math
from help_functions import create_target1
import tensorflow as tf

train_size = 100  # Example: total number of training samples
batch_size = 32  # Example: batch size

steps_for_epoch = math.ceil(train_size / batch_size)
print("Steps for one epoch:", steps_for_epoch)

targets = create_target1(4)
print(tf.gather(targets, 0))
print(tf.gather(targets, 1))
