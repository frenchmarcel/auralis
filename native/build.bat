@echo off
rem Build the Auralis native kernel on Windows.
rem Option 1: "x64 Native Tools Command Prompt for VS", then build.bat
rem Option 2: cmake -S . -B build && cmake --build build --config Release
setlocal
cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 goto try_cmake

set OUT_DIR=..\auralis\lib\windows-x86_64
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
cl /nologo /O2 /EHsc /LD /std:c++17 /fp:precise src\aur_native.cpp src\aur_grid.cpp ^
   /Fe:"%OUT_DIR%\libauralis.dll"
if errorlevel 1 goto fail
del /q aur_native.obj aur_grid.obj 2>nul
del /q "%OUT_DIR%\libauralis.lib" "%OUT_DIR%\libauralis.exp" 2>nul
echo OK -^> %OUT_DIR%\libauralis.dll
goto :eof

:try_cmake
where cmake >nul 2>nul
if errorlevel 1 goto no_tools
cmake -S . -B build || goto fail
cmake --build build --config Release || goto fail
echo OK (via CMake) -^> auralis\lib\windows-*\libauralis.dll
goto :eof

:no_tools
echo Neither cl.exe nor cmake found. Open a "Native Tools Command Prompt"
echo from Visual Studio, or install CMake. The Python fallback works
echo without compiling (slower).
exit /b 1

:fail
echo Build failed.
exit /b 1
