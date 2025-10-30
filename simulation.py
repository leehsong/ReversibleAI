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
    normalized_normals = np.nan_to_num(normalized_normals, nan=0.0, posinf=0.0, neginf=0.0)

    # 오염원이 있는 지점만 추출하여 연산량 감소
    source_indices = np.argwhere(contamination > 0.1)
    source_positions = None
    source_strength = None
    source_normals = None
    if source_indices.size > 0:
        source_y = source_indices[:, 0]
        source_x = source_indices[:, 1]
        source_positions = np.column_stack((
            source_x.astype(float),
            source_y.astype(float),
            terrain[source_y, source_x].astype(float)
        ))
        source_strength = contamination[source_y, source_x].astype(float)
        source_normals = normalized_normals[source_y, source_x, :].astype(float)

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

            if source_positions is None:
                measured_values.append(0.0)
                continue

            vec_to_drone = drone_pos.astype(float) - source_positions  # shape (N, 3)
            dist_sq = np.einsum('ij,ij->i', vec_to_drone, vec_to_drone)
            valid_mask = dist_sq > 1.0

            if not np.any(valid_mask):
                measured_values.append(0.0)
                continue

            vec_to_drone = vec_to_drone[valid_mask]
            dist_sq = dist_sq[valid_mask]
            strength = source_strength[valid_mask]
            normals_subset = source_normals[valid_mask]

            inv_dist = 1.0 / np.sqrt(dist_sq)
            norm_vec_to_drone = vec_to_drone * inv_dist[:, None]
            directionality = np.maximum(
                0.0,
                np.einsum('ij,ij->i', normals_subset, norm_vec_to_drone)
            )

            signal = (strength / dist_sq) * directionality
            total_signal = np.sum(signal)

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


def idw_interpolation(measurements, terrain_shape, power=2.0, smoothing=1e-6, target_scale=None):
    """
    드론 측정값을 이용해 IDW(역거리 가중) 방식으로 오염도를 추정합니다.
    measurements: {'path': [[x, y, z], ...], 'values': [...]}
    terrain_shape: (height, width)
    target_scale: 대상 스케일 상한(예: 실제 오염 맵의 최대값). None이면 입력값 그대로 사용.
    """
    if not measurements or len(measurements.get('path', [])) == 0:
        return np.zeros(terrain_shape, dtype=float)

    coords = np.array([[p[0], p[1]] for p in measurements['path']], dtype=float)
    values = np.array(measurements['values'], dtype=float)

    # 유효한 값 필터링
    valid_mask = np.isfinite(values)
    coords = coords[valid_mask]
    values = values[valid_mask]

    if coords.size == 0:
        return np.zeros(terrain_shape, dtype=float)

    # 측정값을 목표 스케일에 맞춰 정규화 (예: Ground Truth 최대값에 맞춤)
    if target_scale is not None:
        max_val = np.max(np.abs(values))
        if max_val > 0:
            values = values * (target_scale / max_val)

    grid_y, grid_x = np.indices(terrain_shape, dtype=float)
    idw_accumulator = np.zeros(terrain_shape, dtype=float)
    weight_sum = np.zeros(terrain_shape, dtype=float)

    for (x, y), value in zip(coords, values):
        dist_sq = (grid_x - x) ** 2 + (grid_y - y) ** 2
        weights = 1.0 / np.power(dist_sq + smoothing, power / 2.0)
        idw_accumulator += weights * value
        weight_sum += weights

    with np.errstate(divide='ignore', invalid='ignore'):
        interpolated = np.where(weight_sum > 0, idw_accumulator / weight_sum, 0.0)

    # IDW는 측정 지점에서 정확한 값을 갖도록 보정
    for (x, y), value in zip(coords, values):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < terrain_shape[0] and 0 <= xi < terrain_shape[1]:
            interpolated[yi, xi] = value

    return interpolated
