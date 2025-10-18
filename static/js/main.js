// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('terrain-plot')) setupTerrainPage();
    if (document.getElementById('unified-plot')) initContaminationPage();
    if (document.getElementById('simulation-plot')) setupSimulationPage();
    if (document.getElementById('prediction-plot')) setupAiPage();
});

// --- Part 1: 지형 선택 페이지 로직 ---
function setupTerrainPage() {
    const xSizeSlider = document.getElementById('x-size');
    const ySizeSlider = document.getElementById('y-size');
    const xSizeValue = document.getElementById('x-size-value');
    const ySizeValue = document.getElementById('y-size-value');
    const terrainTypeSelect = document.getElementById('terrain-type');
    const generateBtn = document.getElementById('generate-terrain-btn');
    const plotDiv = document.getElementById('terrain-plot');
    const loadingIndicator = document.getElementById('loading-indicator');

    xSizeSlider.addEventListener('input', () => xSizeValue.textContent = xSizeSlider.value);
    ySizeSlider.addEventListener('input', () => ySizeValue.textContent = ySizeSlider.value);

    generateBtn.addEventListener('click', () => {
        plotDiv.style.display = 'none';
        loadingIndicator.style.display = 'block';
        const payload = { x_size: xSizeSlider.value, y_size: ySizeSlider.value, terrain_type: terrainTypeSelect.value };

        fetch('/api/generate_terrain', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) { alert(data.error); return; }
            const layout = { title: '3D 지형', autosize: true, scene: { aspectmode: "data" } };
            const plotData = [{ z: data.z_data, type: 'surface', colorscale: 'Viridis' }];
            Plotly.newPlot(plotDiv, plotData, layout);
        })
        .catch(error => console.error('Error:', error))
        .finally(() => {
            loadingIndicator.style.display = 'none';
            plotDiv.style.display = 'block';
        });
    });
}

// --- Part 2: 오염원 배치 페이지 로직 ---
function initContaminationPage() {
    const plotDiv = document.getElementById('unified-plot');

    const plotData = [{
        z: initialTerrainData,
        surfacecolor: initialContaminationData,
        type: 'surface',
        colorscale: 'Reds',
        cmin: 0,
        cmax: Math.max(10, ...initialContaminationData.flat()),
        colorbar: {title: i18n.colorbarTitle}
    }];

    const layout = {
        title: i18n.plotTitle,
        autosize: true,
        scene: { 
            aspectmode: "data",
            camera: { eye: { x: 1.25, y: 1.25, z: 1.25 } }
        },
        margin: { l: 0, r: 0, b: 0, t: 40 }
    };

    Plotly.newPlot(plotDiv, plotData, layout);

    plotDiv.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            const point = data.points[0];
            const payload = { x: point.x, y: point.y };

            fetch('/api/place_contaminant', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(apiResponse => {
                if (apiResponse.error) { alert(apiResponse.error); return; }
                
                const updatedContamination = apiResponse.contamination_data;
                const newMax = Math.max(10, ...updatedContamination.flat());

                const newData = {
                    ...plotData[0],
                    surfacecolor: updatedContamination,
                    cmax: newMax
                };

                Plotly.react(plotDiv, [newData], layout);
            });
        }
    });
}


// --- Part 3: 드론 시뮬레이션 페이지 로직 (Colormap 통일 기능 추가) ---
function setupSimulationPage() {
    const plotDiv = document.getElementById('simulation-plot');
    const loadingIndicator = document.getElementById('loading-indicator-sim');
    
    // 누적된 모든 측정값을 저장할 배열
    let allMeasuredValues = [];

    // 1. 초기에는 '지형'과 '실제 오염도'를 함께 표시
    const initialLayout = { 
        title: i18n_sim.plotTitle,
        autosize: true, 
        scene: { aspectmode: "data" } 
    };
    const initialPlotData = [{
        z: initialTerrainDataSim, 
        surfacecolor: initialContaminationDataSim,
        type: 'surface', 
        colorscale: 'Reds',
        name: 'Ground Truth', 
        cmin: 0,
        cmax: Math.max(10, ...initialContaminationDataSim.flat()),
        colorbar: {title: i18n_sim.groundTruthColorbarTitle}
    }];
    Plotly.newPlot(plotDiv, initialPlotData, initialLayout);
    
    // '시뮬레이션 시작' 버튼 클릭 이벤트
    document.getElementById('run-simulation-btn').addEventListener('click', () => {
        plotDiv.style.opacity = 0.5;
        loadingIndicator.style.display = 'block';
        
        const altitude = document.getElementById('altitude').value;
        fetch('/api/run_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ altitude: altitude })
        })
        .then(response => response.json())
        .then(data => {
            if (!data.values || data.values.length === 0) return;

            // 2. 새로 수신한 측정값을 누적 배열에 추가
            allMeasuredValues = allMeasuredValues.concat(data.values);
            
            // 3. 누적된 모든 값 중에서 새로운 '전역 최댓값'을 계산
            const globalMax = Math.max(1, ...allMeasuredValues);

            // 4. 새로운 드론 측정값 트레이스(trace)를 정의
            const newDroneTrace = {
                x: data.path.map(p => p[0]),
                y: data.path.map(p => p[1]),
                z: data.path.map(p => p[2]),
                mode: 'markers',
                type: 'scatter3d',
                name: `${i18n_sim.traceName} (Alt: ${altitude})`, // 범례에 고도 표시
                marker: {
                    color: data.values,
                    colorscale: 'Jet',
                    size: 5,
                    showscale: true,
                    cmin: 0,
                    cmax: globalMax, // 전역 최댓값을 cmax로 설정
                    colorbar: { 
                        title: i18n_sim.droneColorbarTitle,
                        x: 1.15
                    }
                }
            };
            
            // 5. 새로운 트레이스를 차트에 추가
            Plotly.addTraces(plotDiv, newDroneTrace).then(() => {
                // 6. 이전에 추가했던 모든 측정값 트레이스를 찾아서 색상 기준을 통일
                const tracesToUpdate = [];
                const traceIndices = [];

                // plotDiv.data는 현재 차트의 모든 트레이스 정보를 담고 있음 (0번은 지표면)
                for (let i = 1; i < plotDiv.data.length; i++) {
                    // 마지막에 추가된 최신 트레이스는 색상 막대를 표시하고, 나머지는 숨김
                    plotDiv.data[i].marker.showscale = (i === plotDiv.data.length - 1);
                    tracesToUpdate.push(plotDiv.data[i]);
                    traceIndices.push(i);
                }

                // 7. Plotly.restyle을 사용해 이전에 그렸던 모든 점들의 cmax 값을 업데이트
                if (tracesToUpdate.length > 1) {
                    const updateData = {
                        'marker.cmax': tracesToUpdate.map(() => globalMax),
                        'marker.showscale': tracesToUpdate.map((_, idx) => idx === tracesToUpdate.length - 1)
                    };
                    Plotly.restyle(plotDiv, updateData, traceIndices);
                }
            });
        })
        .catch(error => console.error('Error:', error))
        .finally(() => {
            loadingIndicator.style.display = 'none';
            plotDiv.style.opacity = 1.0;
        });
    });
}

// --- Part 4: AI 모델 페이지 로직 ---
function setupAiPage() {
    document.getElementById('start-training-btn').addEventListener('click', () => {
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML = i18n_ai.trainingStart;
        
        const payload = {
            terrain_type: document.getElementById('train-terrain-type').value,
            num_samples: document.getElementById('num-samples').value,
            epochs: document.getElementById('epochs').value
        };

        fetch('/api/start_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            statusDiv.innerHTML = `${i18n_ai.trainingComplete} ${data.loss[data.loss.length - 1]}`;
        });
    });

    document.getElementById('predict-btn').addEventListener('click', () => {
        const truthPlot = document.getElementById('ground-truth-plot');
        const predPlot = document.getElementById('prediction-plot');
        truthPlot.innerHTML = i18n_ai.predicting;
        predPlot.innerHTML = i18n_ai.predicting;

        fetch('/api/predict_contamination')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                truthPlot.innerHTML = ''; predPlot.innerHTML = '';
                return;
            }
            
            const layout = { autosize: true, xaxis: { scaleratio: 1 }, yaxis: { scaleratio: 1 } };
            const heatMapData = (zData) => [{ z: zData, type: 'heatmap', colorscale: 'Reds' }];

            Plotly.newPlot(truthPlot, heatMapData(data.ground_truth), { ...layout, title: i18n_ai.groundTruthTitle });
            Plotly.newPlot(predPlot, heatMapData(data.prediction), { ...layout, title: i18n_ai.predictionTitle });
        });
    });
}