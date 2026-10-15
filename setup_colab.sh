#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường tự động cho Google Colab (Linux / T4 GPU)
# Cố định Python 3.10 (isolated venv), PyTorch 2.0.0+cu118, MMCV-Full 1.7.1, NumPy 1.23.5
# ==============================================================================

set -e

echo "=== [1/5] Kiểm tra và cấu hình Python 3.10 cô lập & GPU ==="

VENV_DIR="/opt/py310"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo ">> Đang cài đặt Python 3.10 và tạo môi trường cô lập tại $VENV_DIR..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common build-essential ninja-build
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.10 python3.10-dev python3.10-distutils python3.10-venv

    # Tạo venv không kèm pip hệ thống để tránh lỗi "Cannot uninstall packaging"
    sudo python3.10 -m venv --without-pip "$VENV_DIR"
    sudo chmod -R 777 "$VENV_DIR"

    echo ">> Đang cài đặt pip, setuptools==65.5.1 và wheel vào môi trường Python 3.10..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python" - "pip<24.1" "setuptools==65.5.1" "wheel"
fi

# Gắn symlinks vào /usr/local/bin để mọi cell Colab tự động gọi đúng Python 3.10 này
sudo ln -sf "$VENV_DIR/bin/python" /usr/local/bin/python
sudo ln -sf "$VENV_DIR/bin/python" /usr/local/bin/python3
sudo ln -sf "$VENV_DIR/bin/pip" /usr/local/bin/pip
sudo ln -sf "$VENV_DIR/bin/pip" /usr/local/bin/pip3
hash -r

PY_CMD="$VENV_DIR/bin/python"
PIP_CMD="$VENV_DIR/bin/python -m pip"

echo "Phiên bản Python kích hoạt: $($PY_CMD --version)"
nvidia-smi || echo "Cảnh báo: Không tìm thấy GPU, hệ thống sẽ cố chạy ở chế độ CPU."

echo ">> Đang cài đặt PyTorch 2.0.0+cu118 và NumPy 1.23.5..."
$PIP_CMD install -q \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Khóa cứng numpy==1.23.5 để không bị lỗi np.bool / np.float trong chumpy & smplx
$PIP_CMD install -q "numpy==1.23.5" "setuptools==65.5.1" "cython<3.0.0"

echo "=== [2/5] Đang cài đặt các thư viện toán và mô hình 3D ==="
$PIP_CMD install -q \
    smplx==0.1.28 \
    yacs \
    tensorboardx \
    filterpy \
    chumpy \
    trimesh \
    pyrender \
    json_tricks \
    torchgeometry \
    einops \
    boto3 \
    plyfile \
    timm==0.6.13 \
    pyglet \
    yapf==0.40.1 \
    loguru \
    scipy==1.10.1 \
    scikit-image \
    scikit-learn \
    opencv-python==4.8.0.76 \
    pyyaml \
    tqdm \
    matplotlib \
    matplotlib-inline \
    easydict \
    pycocotools \
    joblib \
    numba

echo "=== [3/5] Đang cài đặt MMCV-Full 1.7.1 ==="
$PIP_CMD install -q https://github.com/camenduru/wheels/releases/download/colab/mmcv_full-1.7.1-cp310-cp310-linux_x86_64.whl

# Tạo symlink tự động cho libcudart.so.12 nếu thiếu trong đường dẫn hệ thống của Colab
CUDART_FILE=$(find /usr/local -name "libcudart.so.12*" 2>/dev/null | head -n 1)
if [ -n "$CUDART_FILE" ]; then
    echo ">> Đã định vị thư viện CUDA 12 tại: $CUDART_FILE"
    sudo ln -sf "$CUDART_FILE" /usr/lib/x86_64-linux-gnu/libcudart.so.12
    sudo ln -sf "$CUDART_FILE" /usr/lib/libcudart.so.12
    sudo ldconfig 2>/dev/null || true
fi

# Tự động lấy đường dẫn thực tế của thư mục chứa setup_colab.sh (dù ở /content hay Drive)
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "=== [4/5] Biên dịch module MMPose & Transformer Utils ==="
cd "$CURRENT_DIR/main/transformer_utils"
$PIP_CMD install -v -e . --no-build-isolation -q
cd "$CURRENT_DIR"

echo "=== [5/5] Biên dịch module đo đạc 3D Mesh-Mesh Intersection ==="
cd "$CURRENT_DIR/data/SHAPY/mesh-mesh-intersection"
$PIP_CMD install -v -e . --no-build-isolation -q
cd "$CURRENT_DIR"

# Khóa lại numpy==1.23.5 phòng trường hợp gói phụ thuộc tự nâng cấp
$PIP_CMD install -q "numpy==1.23.5"

echo "=============================================================================="
echo ">>> CÀI ĐẶT MÔI TRƯỜNG THÀNH CÔNG! BẠN ĐÃ CÓ THỂ CHẠY run_measurement.py <<<"
echo "=============================================================================="