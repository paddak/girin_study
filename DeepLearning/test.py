import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터와 출력 데이터 정의
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 가중치 및 편향 랜덤 초기화
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 선형 모델 정의
def linear_regression(x):
    return W * x + b

# 손실 함수 정의 (평균 제곱 오차)
def mean_square_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 경사 하강법 옵티마이저 정의
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 학습 함수 정의
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_regression(x)
        loss = mean_square_error(y_pred, y)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

# 학습
epochs = 1000
for epoch in range(epochs):
    train_step(X, Y)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {mean_square_error(linear_regression(X), Y).numpy()}")

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
