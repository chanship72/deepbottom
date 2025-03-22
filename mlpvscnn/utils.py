import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os
import time

# MNIST 데이터 다운로드 및 처리 함수
def load_mnist():
    # 파일이 이미 존재하는지 확인
    if os.path.exists('mnist.pkl'):
        with open('mnist.pkl', 'rb') as f:
            return pickle.load(f)
    
    # MNIST 데이터 다운로드
    files = {
        'train_img': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_label': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_img': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_label': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    mnist_data = {}
    
    # 데이터 다운로드 및 처리
    for name, url in files.items():
        filename = url.split('/')[-1]
        print(f"Downloading {filename}...")
        request.urlretrieve(url, filename)
        
        with gzip.open(filename, 'rb') as f:
            if 'img' in name:
                # 이미지 데이터 처리
                f.read(16)  # 헤더 건너뛰기
                buf = f.read()
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                data = data.reshape(-1, 28, 28) / 255.0  # 정규화
            else:
                # 라벨 데이터 처리
                f.read(8)  # 헤더 건너뛰기
                buf = f.read()
                data = np.frombuffer(buf, dtype=np.uint8)
            
            mnist_data[name] = data
        
        # 다운로드한 파일 삭제
        os.remove(filename)
    
    # 데이터 준비
    x_train = mnist_data['train_img']
    t_train = np.zeros((mnist_data['train_label'].size, 10))
    for idx, label in enumerate(mnist_data['train_label']):
        t_train[idx, label] = 1
    
    x_test = mnist_data['test_img']
    t_test = np.zeros((mnist_data['test_label'].size, 10))
    for idx, label in enumerate(mnist_data['test_label']):
        t_test[idx, label] = 1
    
    result = (x_train, t_train, x_test, t_test)
    
    # 처리된 데이터 저장
    with open('mnist.pkl', 'wb') as f:
        pickle.dump(result, f, -1)
    
    return result

# 활성화 함수들
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 오버플로우 방지

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 오버플로우 방지
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

# 손실 함수: 교차 엔트로피
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    delta = 1e-7  # log(0) 방지
    return -np.sum(t * np.log(y + delta)) / batch_size

# 정확도 측정 함수
def calculate_accuracy(x, t, model):
    y = model.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

# 미니배치 생성 함수
def get_mini_batch(x, t, batch_size):
    batch_mask = np.random.choice(x.shape[0], batch_size)
    return x[batch_mask], t[batch_mask]