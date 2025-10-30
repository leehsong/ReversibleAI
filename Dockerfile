# =========================================================
# ReversibleAI - Flask Web Application (port 29000 → 5000)
# =========================================================
FROM python:3.11-slim

# 1. 시스템 패키지 (OpenCV, Pillow 등 런타임 의존성)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 작업 디렉토리
WORKDIR /app

# 3. 소스 복사
COPY . /app

# 4. 파이썬 패키지 설치
RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. 환경 변수 설정
ENV PORT=30000
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
ENV BABEL_DEFAULT_LOCALE=ko
ENV BABEL_DEFAULT_TIMEZONE=Asia/Seoul
ENV SESSION_TYPE=filesystem
ENV SESSION_FILE_DIR=/tmp/sessions

# 6. 포트 노출
EXPOSE 30000

# 7. Flask 실행 (Gunicorn)
CMD ["gunicorn", "-b", "0.0.0.0:30000", "app:app"]
