# FROM python:3.12-slim
FROM pytorch/torchserve:latest

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/best_model/ ./models/best_model/

USER root
RUN pip install --no-cache-dir .[api]
USER 1000

# resnet50's weights:
COPY resnet50_weights_download.py .
RUN python resnet50_weights_download.py

ENV PYTHONPATH=/app

EXPOSE 25565

CMD ["api", "--config", "configs/train_config.yaml"]