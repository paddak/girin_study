import tensorflow as tf
import numpy as np
import matplotlib as plt

#linear regression : y=ax+b
#y(라벨,정답), x(피쳐,학습)을 이용하여 가장 적절한 a(가중치)와 b(편향)을 찾는과정

#1. label, feature 입력
#2. weight, bias 초기화
#3. linear regression 정의
#4. loss function 정의
#5. optimizer 정의
#6. train function 정의
#7. train

#data
x=np.array([1,2,3])
y=np.array([4,5,6])

#weight, bias 초기화
#local minimum방지, 다양한 초기조건에서 학습 시작->다양한 경로 탐색
W=tf.Variable(np.random.randn(),name='weight')
b=tf.Variable(np.random.randn(),name='bias')

#linear regression 정의
def linear_regression(x):
    return W*x+b

#loss function(MSE)
def MSE(y_pred,y_true):
    return tf.reduce_mean(tf.squre(y_pred-y_true))


