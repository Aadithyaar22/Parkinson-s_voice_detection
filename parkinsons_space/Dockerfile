FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements_wav2vec2.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_wav2vec2.txt

COPY . .

RUN mkdir -p uploads && chmod 777 uploads

RUN python scripts/refit_w2v2_local.py

RUN python -c "\
from transformers import AutoFeatureExtractor, AutoModel; \
print('Downloading wav2vec2-XLS-R (1.2 GB)...'); \
AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-300m'); \
AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m'); \
print('Done.')"

EXPOSE 7860

CMD ["python", "app.py"]
