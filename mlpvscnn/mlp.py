import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os
import time
from utils import relu, sigmoid, softmax, cross_entropy_error, calculate_accuracy, get_mini_batch

class MLP:
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.activation_function = relu if activation == 'relu' else sigmoid
        self.params = {}
        self.layers_size = len(hidden_size_list) + 1
        
        # 가중치 초기화
        for i in range(1, self.layers_size + 1):
            if i == 1:
                self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_list[0])
                self.params['b1'] = np.zeros(hidden_size_list[0])
            elif i == self.layers_size:
                self.params[f'W{i}'] = weight_init_std * np.random.randn(hidden_size_list[i-2], output_size)
                self.params[f'b{i}'] = np.zeros(output_size)
            else:
                self.params[f'W{i}'] = weight_init_std * np.random.randn(hidden_size_list[i-2], hidden_size_list[i-1])
                self.params[f'b{i}'] = np.zeros(hidden_size_list[i-1])
        
        self.gradients = {}  # 기울기 저장용
    
    def predict(self, x):
        # 입력 데이터를 1차원으로 펼침
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        for i in range(1, self.layers_size):
            x = np.dot(x, self.params[f'W{i}']) + self.params[f'b{i}']
            x = self.activation_function(x)
        
        # 출력층 계산
        x = np.dot(x, self.params[f'W{self.layers_size}']) + self.params[f'b{self.layers_size}']
        y = softmax(x)
        
        return y
    
    # 손실 함수 계산
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    # 수치 미분을 이용한 기울기 계산
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        
        for i in range(1, self.layers_size + 1):
            self.gradients[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.gradients[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])
            
            # W에 대한 기울기
            for idx in range(self.params[f'W{i}'].size):
                tmp_val = self.params[f'W{i}'].flat[idx]
                
                # f(x+h) 계산
                self.params[f'W{i}'].flat[idx] = tmp_val + 1e-4
                fxh1 = loss_w(self.params)
                
                # f(x-h) 계산
                self.params[f'W{i}'].flat[idx] = tmp_val - 1e-4
                fxh2 = loss_w(self.params)
                
                # 기울기
                self.gradients[f'W{i}'].flat[idx] = (fxh1 - fxh2) / (2*1e-4)
                
                # 값 복원
                self.params[f'W{i}'].flat[idx] = tmp_val
            
            # b에 대한 기울기
            for idx in range(self.params[f'b{i}'].size):
                tmp_val = self.params[f'b{i}'].flat[idx]
                
                # f(x+h) 계산
                self.params[f'b{i}'].flat[idx] = tmp_val + 1e-4
                fxh1 = loss_w(self.params)
                
                # f(x-h) 계산
                self.params[f'b{i}'].flat[idx] = tmp_val - 1e-4
                fxh2 = loss_w(self.params)
                
                # 기울기
                self.gradients[f'b{i}'].flat[idx] = (fxh1 - fxh2) / (2*1e-4)
                
                # 값 복원
                self.params[f'b{i}'].flat[idx] = tmp_val
        
        return self.gradients
    
    # 역전파를 이용한 기울기 계산 (효율성을 위해)
    def backpropagation(self, x, t):
        batch_size = x.shape[0]
        
        # 입력 데이터를 1차원으로 펼침
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        # 순전파
        layer_input = {}  # 각 층의 입력 저장
        layer_output = {}  # 각 층의 활성화 함수 적용 후 출력 저장
        
        layer_input[0] = x
        
        for i in range(1, self.layers_size):
            layer_input[i] = np.dot(layer_output.get(i-1, layer_input[0]), self.params[f'W{i}']) + self.params[f'b{i}']
            layer_output[i] = self.activation_function(layer_input[i])
        
        # 출력층 계산
        final_layer = self.layers_size
        layer_input[final_layer] = np.dot(layer_output[final_layer-1], self.params[f'W{final_layer}']) + self.params[f'b{final_layer}']
        y = softmax(layer_input[final_layer])
        
        # 역전파
        # 출력층 오차
        dy = (y - t) / batch_size
        
        # 출력층 가중치와 편향에 대한 기울기
        self.gradients[f'W{final_layer}'] = np.dot(layer_output[final_layer-1].T, dy)
        self.gradients[f'b{final_layer}'] = np.sum(dy, axis=0)
        
        # 은닉층으로 오차 역전파
        dout = np.dot(dy, self.params[f'W{final_layer}'].T)
        
        for i in range(final_layer-1, 0, -1):
            # ReLU 함수의 미분
            if self.activation_function == relu:
                dx = dout * (layer_output[i] > 0)
            else:  # sigmoid 함수의 미분
                dx = dout * (1.0 - layer_output[i]) * layer_output[i]
            
            # 가중치와 편향에 대한 기울기
            self.gradients[f'W{i}'] = np.dot(layer_output.get(i-1, layer_input[0]).T, dx)
            self.gradients[f'b{i}'] = np.sum(dx, axis=0)
            
            # 이전 층으로 오차 전파
            dout = np.dot(dx, self.params[f'W{i}'].T)
        
        return self.gradients
    
    # 모델 학습
    def train(self, x_train, t_train, x_test, t_test, 
              learning_rate=0.1, epochs=5, batch_size=100, 
              gradient_method='backprop', verbose=True):
        
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        total_iterations = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            for i in range(iter_per_epoch):
                # 미니배치 추출
                batch_x, batch_t = get_mini_batch(x_train, t_train, batch_size)
                
                # 기울기 계산
                if gradient_method == 'numerical':
                    self.numerical_gradient(batch_x, batch_t)
                else:
                    self.backpropagation(batch_x, batch_t)
                
                # 매개변수 갱신
                for j in range(1, self.layers_size + 1):
                    self.params[f'W{j}'] -= learning_rate * self.gradients[f'W{j}']
                    self.params[f'b{j}'] -= learning_rate * self.gradients[f'b{j}']
                
                total_iterations += 1
            
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