import tensorflow as tf
import numpy as np

a=tf.constant(2)
print(a)
b=tf.constant(3.14)
print(b)
c=tf.constant([3,2])
print(c)
print(tf.rank(c))