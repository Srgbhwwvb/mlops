FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/best_model/ ./models/best_model/

RUN pip install --no-cache-dir -e .[api]

# resnet50's weights:
COPY resnet50_weight_download.py .
RUN python resnet50_weight_download.py

ENV PYTHONPATH=/app

EXPOSE 25565

CMD ["api", "--config", "configs/train_config.yaml"]