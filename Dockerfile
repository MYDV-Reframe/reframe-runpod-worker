# CUDA 12.8 + PyTorch 2.7 is required for NVIDIA Blackwell (sm_120) GPUs
# such as RTX PRO 6000. The previous runpod/pytorch:2.4.0-cuda12.4 image
# only shipped kernels through sm_90 and failed RunPod fitness checks with:
#   "no kernel image is available for execution on the device"
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# ImageMagick is required by app/pipeline/compositor.py, which shells out to
# it via subprocess for the vehicle/background composite. Ported from the
# backend's processing/pipeline.py, which has the same dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
        imagemagick \
    && rm -rf /var/lib/apt/lists/* \
    && (test -f /etc/ImageMagick-6/policy.xml \
        && sed -i 's/rights="none" pattern="\(PNG\|JPEG\|PDF\)"/rights="read|write" pattern="\1"/g' \
           /etc/ImageMagick-6/policy.xml || true)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; cuda=torch.version.cuda or ''; assert cuda.startswith('12.8'), f'Expected CUDA 12.8 torch for Blackwell, got {cuda!r} / {torch.__version__}'; print(f'torch={torch.__version__} cuda={cuda}')"

# Bake the segmentation model into the image at build time so cold starts
# don't depend on reaching the model host at runtime. Must match
# PREPROCESSING_SEGMENTATION_MODEL's default (app/settings.py) and
# MODEL_CACHE_DIR/U2NET_HOME below.
ENV U2NET_HOME=/app/models/rembg
RUN mkdir -p /app/models/rembg \
    && python -c "from rembg import new_session; new_session('birefnet-general-lite')"

# Background presets and their images, read by app/pipeline/background_presets.py
COPY assets/ ./assets/

COPY handler.py .
COPY app/ ./app/

CMD ["python", "-u", "handler.py"]