FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the segmentation model into the image at build time so cold starts
# don't depend on reaching the model host at runtime. Must match
# PREPROCESSING_SEGMENTATION_MODEL's default (app/settings.py) and
# MODEL_CACHE_DIR/U2NET_HOME below.
ENV U2NET_HOME=/app/models/rembg
RUN mkdir -p /app/models/rembg \
    && python -c "from rembg import new_session; new_session('birefnet-general-lite')"

COPY handler.py .
COPY app/ ./app/

CMD ["python", "-u", "handler.py"]
