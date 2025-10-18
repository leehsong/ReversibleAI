# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

import terrain_generator
import simulation

# --- 모델 정의 (간단한 U-Net 구조) ---
def build_unet(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    b = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    
    # Decoder
    u1 = UpSampling2D((2, 2))(b)
    u1 = concatenate([u1, c2])
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = UpSampling2D((2, 2))(c3)
    u2 = concatenate([u2, c1])
    c4 = Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    
    outputs = Conv2D(1, (1, 1), activation='relu')(c4) # 오염값은 양수이므로 'relu'
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# --- 학습 데이터 생성 ---
def generate_training_data(terrain, num_samples):
    y_size, x_size = terrain.shape
    X_train = np.zeros((num_samples, y_size, x_size, 1))
    Y_train = np.zeros((num_samples, y_size, x_size, 1))

    for i in range(num_samples):
        print(f"Generating data {i+1}/{num_samples}...")
        # 1. 랜덤 오염원(ground truth) 생성
        ground_truth = np.zeros_like(terrain)
        num_sources = np.random.randint(1, 5)
        for _ in range(num_sources):
            pos_x, pos_y = np.random.randint(0, x_size), np.random.randint(0, y_size)
            strength = np.random.uniform(50, 200)
            sigma = np.random.uniform(3, 8)
            ground_truth = simulation.add_gaussian_source(ground_truth, (pos_y, pos_x), strength, sigma)
        
        # 2. 드론 시뮬레이션으로 측정 데이터(input) 생성
        _, measured_values = simulation.run_drone_scan(terrain, ground_truth, altitude=30)
        drone_path = simulation.run_drone_scan(terrain, ground_truth, altitude=30)[0] # 경로만 다시 가져옴

        sparse_map = np.zeros_like(terrain)
        for idx, p in enumerate(drone_path):
             # 경로 좌표가 맵 범위 내에 있는지 확인
            px, py = int(p[0]), int(p[1])
            if 0 <= px < x_size and 0 <= py < y_size:
                sparse_map[py, px] = measured_values[idx]

        X_train[i, :, :, 0] = sparse_map
        Y_train[i, :, :, 0] = ground_truth
    
    return X_train, Y_train

# --- 모델 학습 및 예측 함수 ---
def train_model_on_terrain(terrain_type, num_samples=100, epochs=20):
    # 1. 학습할 지형 생성
    terrain = terrain_generator.create_terrain(100, 100, terrain_type)
    
    # 2. 학습 데이터 생성
    X_train, Y_train = generate_training_data(terrain, num_samples)
    
    # 3. 모델 빌드 및 학습
    model = build_unet(input_shape=(100, 100, 1))
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=8, validation_split=0.1)
    
    # 4. 학습된 모델 저장
    model.save(f'./models/model_{terrain_type}.h5')
    return history

def predict_with_model(input_data, terrain_type='flat'):
    # 저장된 모델 로드
    # 실제로는 선택된 지형에 맞는 모델을 로드해야 함
    model_path = f'./models/model_{terrain_type}.h5'
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        # 모델이 없을 경우 에러 대신 빈 맵 반환
        return np.zeros(input_data.shape[1:3])

    prediction = model.predict(input_data)
    return prediction[0, :, :, 0]