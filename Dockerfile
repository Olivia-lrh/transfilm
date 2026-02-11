FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      software-properties-common \
      git \
      curl \
      ca-certificates \
      ffmpeg \
      build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3.13 \
      python3.13-venv \
      python3.13-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/

RUN python3.13 -m pip install --upgrade pip \
    && python3.13 -m pip install -e .

COPY src /workspace/src

RUN python3.13 -m pip install flash-attn --no-build-isolation

ENTRYPOINT ["transfilm"]