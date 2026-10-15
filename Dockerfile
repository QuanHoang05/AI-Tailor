# ==============================================================================
# Dockerfile: AI-Tailor - 3D Human Body Measurement Extraction (ISO 8559)
# Base: PyTorch 2.0.1 with CUDA 11.8 Runtime & Python 3.10
# ==============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Cài đặt các gói hệ thống cần thiết cho OpenCV và OpenGL headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Nâng cấp pip và cài đặt setuptools khóa version tương thích
RUN pip install --no-cache-dir "pip<24.1" "setuptools==65.5.1" "wheel" "cython<3.0.0" "numpy==1.23.5"

# Cài đặt MMCV-full 1.7.1 từ wheel tương thích
RUN pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/colab/mmcv_full-1.7.1-cp310-cp310-linux_x86_64.whl

# Cài đặt các thư viện Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy==1.23.5" "matplotlib-inline"

# Copy mã nguồn dự án vào container
COPY . /app

# Cài đặt các module nội bộ ở chế độ develop
RUN cd /app/main/transformer_utils && pip install --no-cache-dir -e . --no-build-isolation
RUN cd /app/data/SHAPY/mesh-mesh-intersection && pip install --no-cache-dir -e . --no-build-isolation

# Khóa lại numpy==1.23.5
RUN pip install --no-cache-dir "numpy==1.23.5"

# Khai báo volume lưu ảnh đầu vào và kết quả đầu ra
VOLUME ["/app/inputs", "/app/outputs"]

ENTRYPOINT ["python", "run_measurement.py"]
CMD ["--help"]
