# simulation.py
import numpy as np
from scipy.ndimage import gaussian_filter

def add_gaussian_source(contamination_map, position, strength=100, sigma=5):
    """
    지표면에 가우시안 형태의 오염원을 '누적'하여 추가합니다.
    contamination_map: 현재 지표면의 오염도 (2D 배열)
    position: 오염원을 추가할 위치 (y, x)
    """
    temp_map = np.zeros_like(contamination_map, dtype=float)
    temp_map[position] = strength
    gaussian_source = gaussian_filter(temp_map, sigma=sigma)
    return contamination_map + gaussian_source

def run_drone_scan(terrain, contamination, altitude):
    """
    드론 스캔을 시뮬레이션하고 측정값을 반환합니다.
    **수정**: 드론이 지형 높이와 상관없이 '절대 고도(Absolute Altitude)'로 비행합니다.
    """
    y_size, x_size = terrain.shape
    
    # 지형의 각 지점에서 법선 벡터(Normal Vector) 계산
    grad_y, grad_x = np.gradient(terrain)
    normals = np.dstack((-grad_x, -grad_y, np.ones_like(terrain)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normalized_normals = normals / norms

    drone_path = []
    measured_values = []
    
    # 드론이 지표면을 스캔
    for y in range(0, y_size, 5):
        scan_line = range(x_size) if y % 2 == 0 else reversed(range(x_size))
        for x in scan_line:
            # ⬇️⬇️⬇️ **핵심 수정 부분** ⬇️⬇️⬇️
            # 드론의 z좌표를 지형 높이와 상관없이 절대 고도(altitude)로 고정
            drone_pos = np.array([x, y, altitude])
            
            # (안전장치) 만약 절대 고도가 지형보다 낮으면(충돌), 해당 지점은 측정하지 않고 건너뜀
            if drone_pos[2] < terrain[y, x]:
                continue
            
            drone_path.append(drone_pos)
            
            total_signal = 0
            # 모든 오염원이 현재 드론 위치에 미치는 영향 계산
            for sy in range(0, y_size, 2):
                for sx in range(0, x_size, 2):
                    if contamination[sy, sx] > 0.1:
                        source_pos = np.array([sx, sy, terrain[sy, sx]])
                        
                        vec_to_drone = drone_pos - source_pos
                        dist_sq = np.sum(vec_to_drone**2)

                        if dist_sq > 1:
                            norm_vec_to_drone = vec_to_drone / np.sqrt(dist_sq)
                            source_normal = normalized_normals[sy, sx]
                            directionality = max(0, np.dot(source_normal, norm_vec_to_drone))
                            
                            signal = (contamination[sy, sx] / dist_sq) * directionality
                            total_signal += signal
            
            measured_values.append(total_signal * 100)
            
    return drone_path, np.array(measured_values)

def format_measurements_for_model(measurements, terrain_shape):
    """측정값을 AI 모델 입력 형식으로 변환합니다."""
    num_measurements = len(measurements['path'])
    input_array = np.zeros(terrain_shape)
    
    for i in range(num_measurements):
        x, y, _ = measurements['path'][i]
        val = measurements['values'][i]
        if 0 <= int(y) < terrain_shape[0] and 0 <= int(x) < terrain_shape[1]:
            input_array[int(y), int(x)] = val

    return np.expand_dims(input_array, axis=(0, -1))