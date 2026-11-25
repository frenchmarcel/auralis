@echo off
rem Build the Auralis CUDA kernel on Windows.
rem
rem Requirements: CUDA Toolkit (>= 12.0 for Ada / RTX 2000 Ada) + Visual
rem Studio. Run this script from an "x64 Native Tools Command Prompt"
rem (nvcc needs cl.exe).
rem
rem   build_cuda.bat            -> auralis_cuda.dll (GPU)
rem   set AUR_CUDA_ARCH=sm_86 & build_cuda.bat   to force the architecture
setlocal
cd /d "%~dp0"

where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc not found: install the CUDA Toolkit and re-run from
  echo an "x64 Native Tools Command Prompt" from Visual Studio.
  exit /b 1
)

set OUT_DIR=..\auralis\lib\windows-x86_64
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set OUT=%OUT_DIR%\auralis_cuda.dll

if not "%AUR_CUDA_ARCH%"=="" (
  nvcc -O3 -std=c++17 --shared -arch=%AUR_CUDA_ARCH% src\aur_cuda.cu -o "%OUT%" && goto done
  goto fail
)

nvcc -O3 -std=c++17 --shared -arch=native src\aur_cuda.cu -o "%OUT%" 2>nul && goto done
nvcc -O3 -std=c++17 --shared -arch=sm_89 src\aur_cuda.cu -o "%OUT%" 2>nul && goto done
nvcc -O3 -std=c++17 --shared -arch=sm_86 src\aur_cuda.cu -o "%OUT%" 2>nul && goto done
nvcc -O3 -std=c++17 --shared -arch=sm_75 src\aur_cuda.cu -o "%OUT%" && goto done

:fail
echo CUDA build failed.
exit /b 1

:done
echo OK -^> %OUT%
echo Check: python -c "from auralis.engine import gpu_available; print(gpu_available())"
