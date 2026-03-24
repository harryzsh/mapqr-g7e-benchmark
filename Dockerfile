FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget ninja-build gcc g++ && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.7 + cu128 (Blackwell native support)
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Build mmcv-full from source with Blackwell arch
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
RUN pip install mmcv-full==1.7.2 --no-build-isolation

# mmdet + mmseg + timm (pinned for compatibility)
RUN pip install mmdet==2.28.2 mmsegmentation==0.30.0 'timm==0.6.13'

# Runtime deps
# NOTE: av2 installed with --no-deps to prevent torch overwrite
# NOTE: numpy<2 pinned LAST to prevent other packages pulling numpy 2.x
RUN pip install nuscenes-devkit==1.1.9 lyft_dataset_sdk \
    opencv-python-headless 'numba>=0.56' \
    universal_pathlib pyarrow pyproj polars \
    networkx plyfile scikit-image shapely einops \
    'matplotlib==3.5.3' trimesh pyquaternion descartes \
    tensorboard IPython && \
    pip install av2 --no-deps && \
    pip install "numpy==1.26.4" --force-reinstall --no-deps
