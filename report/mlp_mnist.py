import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist

# -------------------------------------------------------
# 2. MLP 모델 정의(은닉층 1개)
# -------------------------------------------------------
class MLP:
    def __init__(self, input_dim=(1,28,28), hidden_dim=128, output_dim=10, learning_rate=0.01):
        """
        간단한 MLP 구성:
        input -> (W1, b1) -> hidden -> ReLU -> (W2, b2) -> output -> Softmax

        input_dim은 (1, 28, 28) 형태라고 가정하되,
        내부적으로는 flatten_dim = 1*28*28 = 784 로 펼쳐서 계산합니다.
        """
        self.input_dim = input_dim  # (1,28,28)
        flatten_dim = input_dim[0]*input_dim[1]*input_dim[2]  # 1*28*28 = 784

        # 가중치 초기화
        self.W1 = np.random.randn(flatten_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros((output_dim,), dtype=np.float32)

        self.lr = learning_rate

    def forward(self, x: np.ndarray) -> dict:
        """
        순전파(forward) 계산
        x: (batch_size, 1, 28, 28)
        반환 값: 중간 결과를 모두 dict로 저장 (역전파 때 사용)
        """
        # (batch_size, 1, 28, 28)를 (batch_size, 784)로 펼치기
        batch_size = x.shape[0]
        x_flatten = x.reshape(batch_size, -1)  # (batch_size, 784)

        # hidden = xW1 + b1
        z1 = np.dot(x_flatten, self.W1) + self.b1  # (batch_size, hidden_dim)

        # ReLU
        a1 = np.maximum(z1, 0)  # (batch_size, hidden_dim)

        # output = a1W2 + b2
        z2 = np.dot(a1, self.W2) + self.b2  # (batch_size, output_dim)

        # Softmax
        # exp(z) / sum(exp(z)) 
        z2_shift = z2 - np.max(z2, axis=1, keepdims=True)  # 수치안정
        exp_z2 = np.exp(z2_shift)
        a2 = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

        return {
            'x': x_flatten,  # 펼쳐진 입력 저장
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }

    def backward(self, cache: dict, y_true: np.ndarray):
        """
        역전파(backward) 계산
        cache: forward 수행 시 저장한 중간 결과
        y_true: (batch_size, output_dim) 원-핫 라벨
        """
        x = cache['x']           # (batch_size, 784)
        z1 = cache['z1']         # (batch_size, hidden_dim)
        a1 = cache['a1']         # (batch_size, hidden_dim)
        a2 = cache['a2']         # (batch_size, output_dim)

        batch_size = x.shape[0]

        # ----- 출력층 Softmax + CrossEntropy의 미분 -----
        dz2 = (a2 - y_true) / batch_size  # (batch_size, output_dim)

        # W2, b2의 gradient
        dW2 = np.dot(a1.T, dz2)  # (hidden_dim, output_dim)
        db2 = np.sum(dz2, axis=0)  # (output_dim,)

        # ----- 은닉층 (ReLU) 역전파 -----
        da1 = np.dot(dz2, self.W2.T)      # (batch_size, hidden_dim)
        dz1 = da1 * (z1 > 0)              # ReLU 미분

        # W1, b1의 gradient
        dW1 = np.dot(x.T, dz1)            # (784, hidden_dim)
        db1 = np.sum(dz1, axis=0)        # (hidden_dim,)

        # ----- 매개변수 업데이트 -----
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        교차 엔트로피 손실값 반환
        y_pred: (batch_size, output_dim)
        y_true: (batch_size, output_dim), 원-핫 라벨
        """
        # -sum( y_true * log(y_pred) ) / batch_size
        eps = 1e-7
        batch_size = y_true.shape[0]
        log_likelihood = -np.log(y_pred + eps)
        loss = np.sum(y_true * log_likelihood) / batch_size
        return loss

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        예측값(확률분포 a2)과 실제 라벨(원-핫)로 정확도 계산
        """
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(pred_labels == true_labels)


# -------------------------------------------------------
# 3. 학습 루프
# -------------------------------------------------------
def train(model: MLP, train_x, train_y, test_x, test_y, epochs=5, batch_size=100):
    num_train = train_x.shape[0]
    indices = np.arange(num_train)

    for epoch in range(epochs):
        # 학습 데이터를 섞어서 미니배치 생성
        np.random.shuffle(indices)

        # 미니배치 학습
        for start_idx in range(0, num_train, batch_size):
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]

            x_batch = train_x[batch_idx]  # (batch_size, 1, 28, 28)
            y_batch = train_y[batch_idx]  # (batch_size, 10)

            # forward
            cache = model.forward(x_batch)
            # backward
            model.backward(cache, y_batch)

        # 에폭마다 학습 손실/정확도, 테스트 정확도 출력
        train_cache = model.forward(train_x)
        train_loss = model.loss(train_cache['a2'], train_y)
        train_acc = model.accuracy(train_cache['a2'], train_y)

        test_cache = model.forward(test_x)
        test_acc = model.accuracy(test_cache['a2'], test_y)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")


# -------------------------------------------------------
# 4. 메인 실행부
# -------------------------------------------------------
if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train = x_train[:5000], t_train[:5000]
    x_test, t_test = x_test[:1000], t_test[:1000]

    # # MNIST 데이터 불러오기
    # train_x, train_y, test_x, test_y = load_mnist_dataset()

    # MLP 모델 생성
    mlp = MLP(input_dim=(1, 28, 28), hidden_dim=128, output_dim=10, learning_rate=0.01)
    
    # 학습 수행
    train(model=mlp,
          train_x=x_train, 
          train_y=t_train, 
          test_x=x_test, 
          test_y=t_test, 
          epochs=5,
          batch_size=100)
