import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os
import time
from utils import relu, sigmoid, softmax, cross_entropy_error, calculate_accuracy

class CNN:
    def __init__(self, input_dim=(1, 28, 28), 
                conv_params={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=100, output_size=10, weight_init_std=0.01):
        
        # 가중치 초기화 부분 확인
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_pad = conv_params['pad']
        filter_stride = conv_params['stride']
        
        # 디버깅을 위한 코드 추가
        print("filter_num:", filter_num)
        print("filter_size:", filter_size)
        
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치 초기화        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)        
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        # CNN 구성 저장
        self.input_dim = input_dim
        self.conv_params = conv_params
        self.pool_size = 2  # 풀링 크기 고정
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.gradients = {}  # 기울기 저장
    
    def conv_layer_forward(self, x, W, b, stride=1, pad=0):
        if not isinstance(W, np.ndarray):
            raise ValueError(f"W must be numpy array, but got {type(W)}")
    
        # 디버깅을 위한 코드 추가
        # print("W type:", type(W))
        # print("W shape:", W.shape)
        # print("W value:", W)
        # 입력 형태가 3차원(N, H, W)인 경우 4차원(N, 1, H, W)으로 변환
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            
        FN, C, FH, FW = W.shape
        N, C, H, width = x.shape  # W 대신 width 사용
        out_h = (H + 2*pad - FH) // stride + 1
        out_w = (width + 2*pad - FW) // stride + 1
        
        # 패딩 처리
        if pad > 0:
            x_padded = np.zeros((N, C, H + 2*pad, W + 2*pad))
            x_padded[:, :, pad:pad+H, pad:pad+W] = x
        else:
            x_padded = x
        
        # 결과를 저장할 배열
        col = np.zeros((N, C, FH, FW, out_h, out_w))
        y = np.zeros((N, FN, out_h, out_w))
        
        # 컨볼루션 연산: 간소화된 구현
        for i in range(FH):
            for j in range(FW):
                for s in range(out_h):
                    for t in range(out_w):
                        col[:, :, i, j, s, t] = x_padded[:, :, stride*s+i, stride*t+j]
        
        # 형상 변환
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        W_reshape = W.reshape(FN, -1).T
        
        # 행렬 곱
        out = np.dot(col, W_reshape) + b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        
        return out, col
    
    def max_pooling_forward(self, x, pool_size=2, stride=2):
        N, C, H, W = x.shape
        out_h = (H - pool_size) // stride + 1
        out_w = (W - pool_size) // stride + 1
        
        # 결과를 저장할 배열
        out = np.zeros((N, C, out_h, out_w))
        
        # 최대 풀링 연산
        for i in range(out_h):
            for j in range(out_w):
                x_pool = x[:, :, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                out[:, :, i, j] = np.max(x_pool, axis=(2, 3))
        
        return out
    
    def predict(self, x):
        if x.ndim == 2:  # (N, 784)
            x = x.reshape(x.shape[0], 1, 28, 28)
            
        # 합성곱층 1
        conv1, _ = self.conv_layer_forward(x, self.params['W1'], self.params['b1'], 
                                         self.conv_params['stride'], self.conv_params['pad'])
        relu1 = relu(conv1)
        
        # 풀링층 1
        pool1 = self.max_pooling_forward(relu1, self.pool_size, self.pool_size)
        
        # 완전연결층
        N, C, H, W = pool1.shape
        flatten1 = pool1.reshape(N, -1)
        fc1 = np.dot(flatten1, self.params['W2']) + self.params['b2']
        relu2 = relu(fc1)
        
        # 출력층
        score = np.dot(relu2, self.params['W3']) + self.params['b3']
        
        return softmax(score)
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    # CNN 역전파 구현 (간소화된 버전)
    def backpropagation(self, x, t):
        # 입력 데이터 형태 변환
        if x.ndim == 2:  # (N, 784)
            x = x.reshape(x.shape[0], 1, 28, 28)
        elif x.ndim == 3:  # (N, 28, 28)
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        
        batch_size = x.shape[0]
        
        # 순전파
        # 합성곱층 1
        conv1, col1 = self.conv_layer_forward(x, self.params['W1'], self.params['b1'], 
                                            self.conv_params['stride'], self.conv_params['pad'])
        relu1 = relu(conv1)
        
        # 풀링층 1
        pool1 = self.max_pooling_forward(relu1, self.pool_size, self.pool_size)
        
        # 완전연결층 1
        N, C, H, W = pool1.shape
        flatten1 = pool1.reshape(N, -1)
        fc1 = np.dot(flatten1, self.params['W2']) + self.params['b2']
        relu2 = relu(fc1)
        
        # 출력층
        score = np.dot(relu2, self.params['W3']) + self.params['b3']
        y = softmax(score)
        
        # 역전파
        # 출력층 오차
        dy = (y - t) / batch_size
        
        # W3, b3 기울기
        self.gradients['W3'] = np.dot(relu2.T, dy)
        self.gradients['b3'] = np.sum(dy, axis=0)
        
        # 완전연결층 1 역전파
        drelu2 = np.dot(dy, self.params['W3'].T)
        dfc1 = drelu2 * (fc1 > 0)  # ReLU 미분
        
        # W2, b2 기울기
        self.gradients['W2'] = np.dot(flatten1.T, dfc1)
        self.gradients['b2'] = np.sum(dfc1, axis=0)
        
        # 풀링층 역전파 (간소화)
        dpoolout = np.dot(dfc1, self.params['W2'].T)
        dpool = dpoolout.reshape(N, C, H, W)
        
        # 풀링 -> 컨볼루션 역전파 (간소화)
        dconv = np.zeros_like(conv1)
        pool_size = self.pool_size
        
        for i in range(H):
            for j in range(W):
                # 각 풀링 위치에서 최댓값이 있던 위치에만 기울기 전달
                window = relu1[:, :, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                window_mask = (window == np.max(window, axis=(2, 3), keepdims=True))
                dconv[:, :, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size] = window_mask * dpool[:, :, i:i+1, j:j+1]
        
        # ReLU 역전파
        drelu1 = dconv * (conv1 > 0)
        
        # 컨볼루션 역전파 (간소화)
        FN, C, FH, FW = self.params['W1'].shape
        
        # W1, b1 기울기 계산을 위한 필터 연산
        ddconv = drelu1  # ReLU 통과한 기울기
        
        # 패딩 처리
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        
        # 입력값 패딩
        if pad > 0:
            x_padded = np.zeros((batch_size, C, x.shape[2] + 2*pad, x.shape[3] + 2*pad))
            x_padded[:, :, pad:pad+x.shape[2], pad:pad+x.shape[3]] = x
        else:
            x_padded = x
        
        # W1 기울기 계산 (간소화된 방법)
        self.gradients['W1'] = np.zeros_like(self.params['W1'])
        self.gradients['b1'] = np.sum(ddconv, axis=(0, 2, 3))
        
        for n in range(batch_size):
            for fn in range(FN):
                for i in range(0, x_padded.shape[2] - FH + 1, stride):
                    for j in range(0, x_padded.shape[3] - FW + 1, stride):
                        if i//stride < ddconv.shape[2] and j//stride < ddconv.shape[3]:
                            # 각 필터 위치에 대한 입력값과 기울기 곱
                            self.gradients['W1'][fn] += x_padded[n, :, i:i+FH, j:j+FW] * ddconv[n, fn, i//stride, j//stride]
        
        return self.gradients
    
    # 모델 학습
    def train(self, x_train, t_train, x_test, t_test, 
              learning_rate=0.01, epochs=5, batch_size=100, verbose=True):
        
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            for i in range(iter_per_epoch):
                # 미니배치 추출
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                
                # 기울기 계산
                self.backpropagation(x_batch, t_batch)
                
                # 매개변수 갱신
                for key in self.params.keys():
                    self.params[key] -= learning_rate * self.gradients[key]
            
            # 에폭마다 손실 및 정확도 측정
            loss = self.loss(x_train, t_train)
            train_loss_list.append(loss)
            
            train_acc = calculate_accuracy(x_train, t_train, self)
            train_acc_list.append(train_acc)
            
            test_acc = calculate_accuracy(x_test, t_test, self)
            test_acc_list.append(test_acc)
            
            if verbose:
                print(f"에폭 {epoch+1}/{epochs} | 손실: {loss:.4f} | 훈련 정확도: {train_acc:.4f} | 테스트 정확도: {test_acc:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        return {
            'train_loss': train_loss_list,
            'train_acc': train_acc_list,
            'test_acc': test_acc_list,
            'training_time': training_time
        }
