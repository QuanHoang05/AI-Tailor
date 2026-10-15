@echo off
chcp 65001 >nul
echo ==============================================================================
echo       AI-TAILOR: CÀI ĐẶT MÔI TRƯỜNG TỰ ĐỘNG TRÊN WINDOWS (PYTHON 3.10)
echo ==============================================================================

:: Kiểm tra phiên bản Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Không tìm thấy Python! Vui lòng cài đặt Python 3.10 (64-bit) và tích chọn "Add Python to PATH".
    pause
    exit /b 1
)

echo [*] Đang nâng cấp pip và cài đặt setuptools khóa version tương thích...
python -m pip install --upgrade "pip<24.1" "setuptools==65.5.1" "wheel" "cython<3.0.0" "numpy==1.23.5"

echo [*] Đang cài đặt PyTorch 2.0.1 CUDA 11.8...
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo [*] Đang cài đặt danh sách thư viện từ requirements.txt...
python -m pip install -r requirements.txt
python -m pip install "numpy==1.23.5" "matplotlib-inline"

echo [*] Đang liên kết các module nội bộ...
cd main\transformer_utils
python -m pip install -e . --no-build-isolation -q
cd ..\..

cd data\SHAPY\mesh-mesh-intersection
python -m pip install -e . --no-build-isolation -q
cd ..\..\..

:: Khóa lại numpy==1.23.5
python -m pip install -q "numpy==1.23.5"

echo [*] Đang kiểm tra tính toàn vẹn của các file checkpoints...
python download_models.py

echo ==============================================================================
echo [✓] CÀI ĐẶT HOÀN TẤT! Bạn có thể chạy đo thử nghiệm bằng lệnh:
echo     python run_measurement.py --image ..\..\imageTest\ten_anh.jpg --height_cm 163 --weight_kg 64
echo ==============================================================================
pause
