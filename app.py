# app.py
from flask import Flask, render_template, request, jsonify, session, url_for, redirect, g
from flask_session import Session
from flask_babel import Babel, gettext as _
import numpy as np
import pandas as pd
import os
import json

# 사용자 정의 Python 모듈 임포트
import terrain_generator
import simulation
import model

app = Flask(__name__)

# --- 세션 설정 ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# --- 세션 설정 완료 ---

# --- 다국어(Babel) 설정 ---
app.config['LANGUAGES'] = {'en': 'English', 'ja': 'Japanese', 'ko': 'Korean'}
app.config['BABEL_DEFAULT_LOCALE'] = 'ko'

def get_locale():
    """g.lang_code에 저장된 언어 코드를 반환합니다."""
    return getattr(g, 'lang_code', app.config['BABEL_DEFAULT_LOCALE'])

babel = Babel(app, locale_selector=get_locale)
# --- 다국어 설정 완료 ---


# ⬇️⬇️⬇️ **핵심 수정: 새로운 @app.before_request 핸들러** ⬇️⬇️⬇️
@app.before_request
def before_request_func():
    """
    모든 요청이 view 함수로 전달되기 직전에 실행됩니다.
    URL 경로에서 언어 코드를 '읽어서' g.lang_code에 저장합니다.
    이 방식은 URL 인수를 제거하지 않으므로, view 함수가 인수를 정상적으로 받을 수 있습니다.
    """
    if request.view_args and 'lang_code' in request.view_args:
        # URL에 lang_code가 있고, 지원하는 언어 목록에 있으면 g에 저장
        if request.view_args['lang_code'] in app.config['LANGUAGES']:
            g.lang_code = request.view_args['lang_code']
        else:
            # 유효하지 않은 언어 코드일 경우 기본값(ko) 사용
            g.lang_code = app.config['BABEL_DEFAULT_LOCALE']
    else:
        # URL에 lang_code가 없는 경우 (예: /api/...), 기본값 사용
        g.lang_code = app.config['BABEL_DEFAULT_LOCALE']

# ⬇️⬇️⬇️ **핵심 수정: url_for가 언어 코드를 자동으로 추가하도록 설정** ⬇️⬇️⬇️
@app.url_defaults
def add_language_code(endpoint, values):
    """url_for() 함수 사용 시 자동으로 URL에 언어 코드를 포함시킵니다."""
    if 'lang_code' not in values:
        values['lang_code'] = getattr(g, 'lang_code', app.config['BABEL_DEFAULT_LOCALE'])

# --- URL 규칙 (모든 경로에 언어 코드 추가) ---
@app.route('/', methods=['GET', 'POST'])
@app.route('/<lang_code>/', methods=['GET', 'POST'])
def index(lang_code=None):
    if request.method == 'POST':
        session['username'] = request.form['username']
        # 이제 lang_code를 명시하지 않아도 자동으로 추가됨
        return redirect(url_for('main_menu'))
    if 'username' in session:
        return redirect(url_for('main_menu'))
    return render_template('index.html')

@app.route('/<lang_code>/logout')
def logout(lang_code):
    session.clear()
    return redirect(url_for('index'))

@app.route('/<lang_code>/main')
def main_menu(lang_code):
    if 'username' not in session: return redirect(url_for('index'))
    return render_template('main.html', username=session.get('username'))

@app.route('/<lang_code>/terrain_selection')
def terrain_selection(lang_code):
    if 'username' not in session: return redirect(url_for('index'))
    return render_template('part1_terrain.html')

@app.route('/<lang_code>/contaminant_placement')
def contaminant_placement(lang_code):
    if 'username' not in session: return redirect(url_for('index'))
    if 'terrain_data' not in session: return redirect(url_for('terrain_selection'))
    terrain_data = session.get('terrain_data')
    contamination_data = session.get('contamination_data', np.zeros_like(np.array(terrain_data)).tolist())
    return render_template('part2_contaminant.html',
        terrain_data_json=json.dumps(terrain_data),
        contamination_data_json=json.dumps(contamination_data))

@app.route('/<lang_code>/drone_simulation')
def drone_simulation(lang_code):
    if 'username' not in session: return redirect(url_for('index'))
    if 'contamination_data' not in session: return redirect(url_for('contaminant_placement'))
    return render_template('part3_simulation.html',
        terrain_data_json=json.dumps(session.get('terrain_data')),
        contamination_data_json=json.dumps(session.get('contamination_data')))

@app.route('/<lang_code>/ai_model')
def ai_model(lang_code):
    if 'username' not in session: return redirect(url_for('index'))
    return render_template('part4_training.html')

# --- API 라우트는 언어 코드 없이 그대로 둡니다 ---
@app.route('/api/generate_terrain', methods=['POST'])
def generate_terrain_api():
    data = request.json
    x_size = int(data.get('x_size', 100))
    y_size = int(data.get('y_size', 100))
    terrain_type = data.get('terrain_type', 'flat')

    if terrain_type.startswith('dem'):
        file_path = os.path.join('static', 'data', f'{terrain_type}.csv')
        try:
            z_data = pd.read_csv(file_path, header=None).values
            z_data = z_data[:y_size, :x_size]
        except FileNotFoundError:
            return jsonify({'error': _('%(filename)s 파일을 찾을 수 없습니다.', filename=f'{terrain_type}.csv')}), 404
    else:
        z_data = terrain_generator.create_terrain(x_size, y_size, terrain_type)

    session['terrain_data'] = z_data.tolist()
    session['terrain_type'] = terrain_type
    session.pop('contamination_data', None)
    session.pop('drone_measurements', None)
    return jsonify({'z_data': z_data.tolist()})

@app.route('/api/place_contaminant', methods=['POST'])
def place_contaminant_api():
    if 'terrain_data' not in session:
        return jsonify({'error': _('지형 데이터가 없습니다. 먼저 지형을 생성하세요.')}), 400

    click_data = request.json
    x_click, y_click = int(click_data['x']), int(click_data['y'])
    
    existing_contamination = session.get('contamination_data')

    if existing_contamination:
        contamination_map = np.array(existing_contamination)
    else:
        terrain_data = np.array(session.get('terrain_data'))
        contamination_map = np.zeros_like(terrain_data)

    if 0 <= y_click < contamination_map.shape[0] and 0 <= x_click < contamination_map.shape[1]:
        updated_contamination = simulation.add_gaussian_source(
            contamination_map, (y_click, x_click), strength=100, sigma=5)
        
        session['contamination_data'] = updated_contamination.tolist()
        session.modified = True
        
        print(f"Contamination added: total = {np.sum(updated_contamination):.2f}")
        
        return jsonify({'contamination_data': updated_contamination.tolist()})
    else:
        return jsonify({'error': _('클릭 좌표가 범위를 벗어났습니다.')}), 400


@app.route('/api/random_contaminants', methods=['POST'])
def random_contaminants_api():
    if 'terrain_data' not in session:
        return jsonify({'error': _('지형 데이터가 없습니다. 먼저 지형을 생성하세요.')}), 400

    terrain = np.array(session.get('terrain_data'))
    contamination = np.zeros_like(terrain, dtype=float)

    payload = request.json or {}
    min_count = int(payload.get('min_count', 1))
    max_count = int(payload.get('max_count', 20))
    if min_count < 1:
        min_count = 1
    if max_count < min_count:
        max_count = min_count

    rng = np.random.default_rng()
    count = int(payload.get('count', rng.integers(min_count, max_count + 1)))

    strength_min, strength_max = float(payload.get('strength_min', 50)), float(payload.get('strength_max', 200))
    sigma_min, sigma_max = float(payload.get('sigma_min', 3)), float(payload.get('sigma_max', 8))

    height, width = terrain.shape
    for _ in range(count):
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        strength = float(payload.get('strength')) if 'strength' in payload else float(rng.uniform(strength_min, strength_max))
        sigma = float(payload.get('sigma')) if 'sigma' in payload else float(rng.uniform(sigma_min, sigma_max))
        contamination = simulation.add_gaussian_source(contamination, (y, x), strength=strength, sigma=sigma)

    session['contamination_data'] = contamination.tolist()
    session.modified = True
    return jsonify({'contamination_data': contamination.tolist()})

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation_api():
    altitude = float(request.json.get('altitude', 30))
    terrain = np.array(session.get('terrain_data'))
    contamination = np.array(session.get('contamination_data'))
    drone_path, measured_values = simulation.run_drone_scan(terrain, contamination, altitude)
    session['drone_measurements'] = {'path': [p.tolist() for p in drone_path], 'values': measured_values.tolist()}
    session.modified = True
    return jsonify(session['drone_measurements'])

@app.route('/api/start_training', methods=['POST'])
def start_training_api():
    config = request.json
    history, was_fine_tuned = model.train_model_on_terrain(
        terrain_type=config['terrain_type'],
        num_samples=int(config['num_samples']),
        epochs=int(config['epochs']),
        reuse_existing=bool(config.get('reuse_existing', True)))
    return jsonify({
        "status": "Training Complete",
        "loss": history.history['loss'],
        "mode": "fine_tuned" if was_fine_tuned else "fresh"
    })

@app.route('/api/predict_contamination', methods=['GET'])
def predict_contamination_api():
    if 'drone_measurements' not in session:
        return jsonify({"error": _('예측에 사용할 드론 데이터가 없습니다. 3단계 시뮬레이션을 먼저 실행하세요.')}), 400
    measurements = session.get('drone_measurements')
    terrain_data = session.get('terrain_data')
    terrain_shape = np.array(terrain_data).shape
    ground_truth = session.get('contamination_data')
    ground_truth_array = np.array(ground_truth) if ground_truth is not None else None
    target_scale = float(np.max(ground_truth_array)) if ground_truth_array is not None else None
    terrain_type = session.get('terrain_type', 'flat')
    input_data = simulation.format_measurements_for_model(measurements, terrain_shape)
    try:
        predicted_map = model.predict_with_model(input_data, terrain_type=terrain_type)
    except Exception as exc:
        return jsonify({"error": _('AI 예측 중 오류가 발생했습니다: %(message)s', message=str(exc))}), 500
    idw_map = simulation.idw_interpolation(measurements, terrain_shape, target_scale=target_scale)
    return jsonify({
        'ground_truth': ground_truth,
        'prediction': predicted_map.tolist(),
        'idw_prediction': idw_map.tolist(),
        'terrain_type': terrain_type
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=30000)
