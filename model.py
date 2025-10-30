# model.py
import os
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
    
    # 최종 출력은 음수가 될 수도 있지만 후처리에서 0으로 보정합니다.
    outputs = Conv2D(1, (1, 1), activation='linear')(c4)
    
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
        drone_path, measured_values = simulation.run_drone_scan(terrain, ground_truth, altitude=30)

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
def train_model_on_terrain(terrain_type, num_samples=100, epochs=20, reuse_existing=True):
    # 1. 학습할 지형 생성
    terrain = terrain_generator.create_terrain(100, 100, terrain_type)
    
    # 2. 학습 데이터 생성
    X_train, Y_train = generate_training_data(terrain, num_samples)
    
    # 3. 모델 로드 또는 새로 빌드
    model_path = f'./models/model_{terrain_type}.h5'
    model = None
    fine_tuned = False

    if reuse_existing and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer='adam', loss='mean_squared_error')
            fine_tuned = True
            print(f"[model] Loaded existing model for terrain '{terrain_type}'. Continuing training.")
        except Exception as exc:
            print(f"[model] Failed to load existing model at {model_path}: {exc}. Training from scratch.")
            model = None

    if model is None:
        model = build_unet(input_shape=(100, 100, 1))
        print(f"[model] Building new model for terrain '{terrain_type}'.")
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=8, validation_split=0.1)
    
    # 4. 학습된 모델 저장
    model.save(model_path)
    return history, fine_tuned

def predict_with_model(input_data, terrain_type='flat'):
    # 저장된 모델 로드
    # 실제로는 선택된 지형에 맞는 모델을 로드해야 함
    model_path = f'./models/model_{terrain_type}.h5'
    try:
        model = tf.keras.models.load_model(model_path)
    except (OSError, IOError) as exc:
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}. '{terrain_type}' 지형에 대해 먼저 모델을 학습하세요."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"모델을 불러오는 중 오류가 발생했습니다: {exc}"
        ) from exc

    # 모델이 기대하는 입력 크기와 실제 지형 크기가 다르면 동적으로 보정
    input_height, input_width = input_data.shape[1:3]
    expected_height = model.input_shape[1]
    expected_width = model.input_shape[2]

    resized_input = input_data.astype(np.float32)
    if expected_height and expected_width:
        if input_height != expected_height or input_width != expected_width:
            resized_tensor = tf.image.resize(
                tf.convert_to_tensor(resized_input),
                (expected_height, expected_width),
                method='bilinear'
            )
            resized_input = resized_tensor.numpy()

    prediction = model.predict(resized_input, verbose=0)

    # 예측 결과를 다시 원본 지형 크기로 되돌림
    if prediction.shape[1] != input_height or prediction.shape[2] != input_width:
        prediction = tf.image.resize(
            prediction,
            (input_height, input_width),
            method='bilinear'
        ).numpy()

    # 최종 출력은 음수일 수 있으므로 0 이하를 잘라냅니다.
    return np.maximum(prediction[0, :, :, 0], 0)
