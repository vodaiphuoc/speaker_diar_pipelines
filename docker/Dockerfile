FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

LABEL org.opencontainers.image.ref.name="development" \
      org.opencontainers.image.description="Development and testing image; not production-ready"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements/ ./requirements/

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements/dev.txt

COPY SDP/ ./SDP/
COPY configs/ ./configs/
COPY exports/ ./exports/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY run_phase_three.py ./

CMD ["/bin/bash"]
