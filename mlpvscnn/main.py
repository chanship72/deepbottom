import sys, os
sys.path.append("/Users/chanshinpark/workspace/deepbottom")  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from utils import calculate_accuracy
from common.simpleConvNet import SimpleConvNet
from mlp import MLP
from cnn import CNN
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
fontprop = fm.FontProperties(fname=path, size=12)

# MNIST 데이터 로드 함수 (이전 코드와 동일)
def load_mnist():
    # 파일이 이미 존재하는지 확인
    if os.path.exists('mnist.pkl'):
        with open('mnist.pkl', 'rb') as f:
            return pickle.load(f)
    
    # MNIST 데이터 다운로드
    files = {
        'train_img': 'dataset/mnist_data/train-images-idx3-ubyte.gz',
        'train_label': 'dataset/mnist_data//train-labels-idx1-ubyte.gz',
        'test_img': 'dataset/mnist_data//t10k-images-idx3-ubyte.gz',
        'test_label': 'dataset/mnist_data//t10k-labels-idx1-ubyte.gz'
    }
    
    mnist_data = {}
    
    # 데이터 다운로드 및 처리
    for name, url in files.items():
        filename = url
        # print(f"Downloading {filename}...")
        # request.urlretrieve(url, filename)
        
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

# MLP 및 CNN 모델 클래스 (이전 코드와 동일)
# 여기에는 MLP, CNN 클래스와 관련 함수들이 포함됩니다.
# 위의 코드에서 가져온 것으로 간주합니다.

# 실험 1: 훈련 데이터 크기에 따른 MLP와 CNN의 성능 비교
def experiment_1(x_train, t_train, x_test, t_test):
    print("\n=== 실험 1: 훈련 데이터 크기에 따른 MLP vs CNN 성능 비교 ===")
    # x_train이 (N, 28, 28) 형태인 경우 (N, 1, 28, 28) 또는 (N, 784)로 변환
    if x_train.ndim == 3:
        # CNN용 형태로 유지하거나 MLP용으로 변환
        x_train_mlp = x_train.reshape(x_train.shape[0], -1)  # (N, 784)
    else:
        x_train_mlp = x_train
        
    # 훈련 데이터 크기 설정
    train_sizes = [1000, 5000, 10000, 20000, 50000]
    mlp_accuracies = []
    cnn_accuracies = []
    mlp_times = []
    cnn_times = []
    
    for size in train_sizes:
        # 데이터 샘플링
        indices = np.random.choice(x_train.shape[0], size, replace=False)
        x_train_sample = x_train[indices]
        t_train_sample = t_train[indices]
        
        print(f"\n훈련 데이터 크기: {size}")
        
        # MLP 모델 훈련
        mlp = MLP(input_size=784, hidden_size_list=[5], output_size=10)
        mlp_result = mlp.train(x_train_sample, t_train_sample, x_test, t_test, 
                               learning_rate=0.1, epochs=5, batch_size=100, verbose=True)
        
        # CNN 모델 훈련
        conv_params = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}
        cnn = CNN(input_dim=(1, 28, 28), conv_params=conv_params, hidden_size=100, output_size=10)
        cnn_result = cnn.train(x_train_sample, t_train_sample, x_test, t_test, 
                               learning_rate=0.01, epochs=10, batch_size=100, verbose=True)
        
        # 결과 저장
        mlp_accuracies.append(mlp_result['test_acc'][-1])
        cnn_accuracies.append(cnn_result['test_acc'][-1])
        mlp_times.append(mlp_result['training_time'])
        cnn_times.append(cnn_result['training_time'])
        
        print(f"MLP 정확도: {mlp_result['test_acc'][-1]:.4f}, 학습 시간: {mlp_result['training_time']:.2f}초")
        print(f"CNN 정확도: {cnn_result['test_acc'][-1]:.4f}, 학습 시간: {cnn_result['training_time']:.2f}초")
    
    # 결과 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, mlp_accuracies, marker='o', label='MLP')
    plt.plot(train_sizes, cnn_accuracies, marker='s', label='CNN')
    plt.title('훈련 데이터 크기에 따른 테스트 정확도')
    plt.xlabel('훈련 데이터 크기')
    plt.ylabel('정확도')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, mlp_times, marker='o', label='MLP')
    plt.plot(train_sizes, cnn_times, marker='s', label='CNN')
    plt.title('훈련 데이터 크기에 따른 학습 시간')
    plt.xlabel('훈련 데이터 크기')
    plt.ylabel('시간 (초)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment1_results.png')
    plt.show()
    
    return {
        'train_sizes': train_sizes,
        'mlp_accuracies': mlp_accuracies,
        'cnn_accuracies': cnn_accuracies,
        'mlp_times': mlp_times,
        'cnn_times': cnn_times
    }

# 실험 2: 회전 변환에 대한 MLP와 CNN의 견고성 비교
def experiment_2(x_train, t_train, x_test, t_test):
    print("\n=== 실험 2: 회전 변환에 대한 MLP vs CNN 견고성 비교 ===")
    
    # 회전 각도 설정
    rotation_angles = [0, 5, 10, 15, 20, 25, 30]
    mlp_accuracies = []
    cnn_accuracies = []
    
    # 테스트 데이터 크기 제한 (실험 속도를 위해)
    test_size = 2000
    indices = np.random.choice(x_test.shape[0], test_size, replace=False)
    x_test_sample = x_test[indices]
    t_test_sample = t_test[indices]
    
    # 모델 훈련 (회전되지 않은 원본 데이터로)
    print("원본 데이터로 모델 훈련 중...")
    
    # MLP 모델 훈련
    mlp = MLP(input_size=784, hidden_size_list=[100], output_size=10)
    mlp.train(x_train, t_train, x_test, t_test, 
              learning_rate=0.1, epochs=5, batch_size=100, verbose=True)
    
    # CNN 모델 훈련
    # conv_params = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}
    # cnn = CNN(input_dim=(1, 28, 28), conv_params=conv_params, hidden_size=100, output_size=10)
    # cnn.train(x_train, t_train, x_test, t_test, 
    #           learning_rate=0.01, epochs=5, batch_size=100, verbose=True)
    
    cnn = SimpleConvNet()
    cnn.load_params('chapter7/params_self.pkl')
    
    # 회전 변환을 적용한 테스트 데이터로 평가
    for angle in rotation_angles:
        # 회전 변환 적용
        x_test_rotated = rotate_images(x_test_sample, angle)
        
        # 정확도 측정
        mlp_acc = calculate_accuracy(x_test_rotated, t_test_sample, mlp)
        cnn_acc = calculate_accuracy(x_test_rotated, t_test_sample, cnn)
        
        mlp_accuracies.append(mlp_acc)
        cnn_accuracies.append(cnn_acc)
        
        print(f"회전 각도: {angle}°, MLP 정확도: {mlp_acc:.4f}, CNN 정확도: {cnn_acc:.4f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(rotation_angles, mlp_accuracies, marker='o', label='MLP')
    plt.plot(rotation_angles, cnn_accuracies, marker='s', label='CNN')
    plt.title('회전 변환에 대한 모델 견고성 비교')
    plt.xlabel('회전 각도 (도)')
    plt.ylabel('정확도')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment2_results.png')
    plt.show()
    
    # 회전된 이미지 예시 시각화
    plt.figure(figsize=(15, 5))
    for i, angle in enumerate([0, 10, 20, 30]):
        plt.subplot(1, 4, i+1)
        img_idx = 0  # 첫 번째 이미지
        rotated_img = rotate_images(x_test_sample[img_idx:img_idx+1], angle)[0]
        plt.imshow(rotated_img.reshape(28, 28), cmap='gray')
        plt.title(f'회전 각도: {angle}°')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('rotation_examples.png')
    plt.show()
    
    return {
        'rotation_angles': rotation_angles,
        'mlp_accuracies': mlp_accuracies,
        'cnn_accuracies': cnn_accuracies
    }

# 이미지 회전 함수
def rotate_images(images, angle):
    # 이미지 배열 복사
    rotated_images = np.zeros_like(images)
    
    # 각 이미지에 회전 적용
    for i in range(len(images)):
        # 회전 행렬 생성
        radians = np.radians(angle)
        cos_val = np.cos(radians)
        sin_val = np.sin(radians)
        rotation_matrix = np.array([
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ])
        
        # 이미지 중심 좌표
        center_x, center_y = 14, 14  # 28x28 이미지의 중심
        
        # 회전된 이미지 생성
        rotated_image = np.zeros((28, 28))
        for y in range(28):
            for x in range(28):
                # 중심 기준 좌표 계산
                coord = np.array([x - center_x, y - center_y])
                # 회전 적용
                rotated_coord = np.dot(rotation_matrix, coord) + np.array([center_x, center_y])
                
                # 회전된 좌표가 이미지 내에 있는지 확인
                if (0 <= rotated_coord[0] < 28 and 0 <= rotated_coord[1] < 28):
                    # 가장 가까운 픽셀 가져오기 (nearest neighbor)
                    rx, ry = int(rotated_coord[0]), int(rotated_coord[1])
                    rotated_image[y, x] = images[i, ry, rx]
        
        rotated_images[i] = rotated_image
    
    return rotated_images

# 실험 3: 노이즈에 대한 MLP와 CNN의 견고성 비교
def experiment_3(x_train, t_train, x_test, t_test):
    print("\n=== 실험 3: 노이즈에 대한 MLP vs CNN 견고성 비교 ===")
    
    # 노이즈 레벨 설정
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    mlp_accuracies = []
    cnn_accuracies = []
    
    # 테스트 데이터 크기 제한 (실험 속도를 위해)
    test_size = 2000
    indices = np.random.choice(x_test.shape[0], test_size, replace=False)
    x_test_sample = x_test[indices]
    t_test_sample = t_test[indices]
    
    # 모델
# 모델 훈련 (노이즈가 없는 원본 데이터로)
    print("원본 데이터로 모델 훈련 중...")
    
    # MLP 모델 훈련
    mlp = MLP(input_size=784, hidden_size_list=[100], output_size=10)
    mlp.train(x_train, t_train, x_test, t_test, 
              learning_rate=0.1, epochs=10, batch_size=100, verbose=True)
    
    # CNN 모델 훈련
    # conv_params = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}
    # cnn = CNN(input_dim=(1, 28, 28), conv_params=conv_params, hidden_size=100, output_size=10)
    # cnn.train(x_train, t_train, x_test, t_test, 
    #           learning_rate=0.01, epochs=5, batch_size=100, verbose=True)
    
    cnn = CNN()
    cnn.load_params('chapter7/params_self.pkl')

    
    # 노이즈를 적용한 테스트 데이터로 평가
    for noise_level in noise_levels:
        # 노이즈 적용
        x_test_noisy = add_noise(x_test_sample, noise_level)
        
        # 정확도 측정
        mlp_acc = calculate_accuracy(x_test_noisy, t_test_sample, mlp)
        cnn_acc = calculate_accuracy(x_test_noisy, t_test_sample, cnn)
        
        mlp_accuracies.append(mlp_acc)
        cnn_accuracies.append(cnn_acc)
        
        print(f"노이즈 레벨: {noise_level:.1f}, MLP 정확도: {mlp_acc:.4f}, CNN 정확도: {cnn_acc:.4f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, mlp_accuracies, marker='o', label='MLP')
    plt.plot(noise_levels, cnn_accuracies, marker='s', label='CNN')
    plt.title('노이즈에 대한 모델 견고성 비교')
    plt.xlabel('노이즈 레벨')
    plt.ylabel('정확도')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment3_results.png')
    plt.show()
    
    # 노이즈 적용 이미지 예시 시각화
    plt.figure(figsize=(15, 5))
    for i, noise in enumerate([0, 0.2, 0.3, 0.5]):
        plt.subplot(1, 4, i+1)
        img_idx = 0  # 첫 번째 이미지
        noisy_img = add_noise(x_test_sample[img_idx:img_idx+1], noise)[0]
        plt.imshow(noisy_img.reshape(28, 28), cmap='gray')
        plt.title(f'노이즈 레벨: {noise:.1f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('noise_examples.png')
    plt.show()
    
    return {
        'noise_levels': noise_levels,
        'mlp_accuracies': mlp_accuracies,
        'cnn_accuracies': cnn_accuracies
    }

# 이미지에 노이즈 추가 함수
def add_noise(images, noise_level):
    # 이미지 배열 복사
    noisy_images = images.copy()
    
    # 노이즈 추가
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, images.shape)
        noisy_images = noisy_images + noise
        
        # 픽셀 값 범위 조정 (0-1 사이)
        noisy_images = np.clip(noisy_images, 0, 1)
    
    return noisy_images

# 메인 실험 함수
def run_experiments():
    x_train, t_train, x_test, t_test = load_mnist()
    
    # 실험 1: 훈련 데이터 크기에 따른 성능 비교
    experiment1_results = experiment_1(x_train, t_train, x_test, t_test)
    
    # 실험 2: 회전 변환에 대한 견고성 비교
    experiment2_results = experiment_2(x_train, t_train, x_test, t_test)
    
    # 실험 3: 노이즈에 대한 견고성 비교
    experiment3_results = experiment_3(x_train, t_train, x_test, t_test)

if __name__ == "__main__":
    run_experiments()