FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-lock.txt ./requirements-lock.txt
RUN pip install -r requirements-lock.txt
COPY ml ./ml
COPY backend ./backend
COPY emulator ./emulator
COPY tests ./tests
COPY models ./models
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
