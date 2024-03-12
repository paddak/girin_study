import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
X=np.array([np.random.randint(),np.random.randint(),np.random.randint()])
Y=np.array([4,5,6])

#weight, bias 초기화
#local minimum방지, 다양한 초기조건에서 학습 시작->다양한 경로 탐색
W=tf.Variable(np.random.randn(),name='weight')
b=tf.Variable(np.random.randn(),name='bias')

#linear regression 정의
def linear_regression(x):
    return W*x+b

#loss function(MSE)
def MSE(y_pred,y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))

#optimizer 정의
optimizer=tf.optimizers.SGD(learning_rate=0.01)

#train function 정의
def train_function(x,y):
    #gradient기록
    with tf.GradientTape() as tape:
        y_pred=linear_regression(x)
        loss=MSE(y_pred,y)
    
    gradients = tape.gradient(loss, [W,b])
    optimizer.apply_gradients(zip(gradients, [W,b]))

#train
epochs=1000
for epoch in range(epochs):
    train_function(X,Y)
    if epoch % 100==0:
        print(f"Epoch {epoch}, Loss:{MSE(linear_regression(X),Y).numpy()}")

# 학습된 가중치 및 편향 출력
print("학습된 가중치:", W.numpy())
print("학습된 편향:", b.numpy())

# 시각화
plt.scatter(X, Y, label='Data')
plt.plot(X, linear_regression(X), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()