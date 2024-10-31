@echo off
setlocal enabledelayedexpansion

echo Building C++ visualization module...

:: Clean and create build directory
if exist build rmdir /s /q build
mkdir build
cd build

:: Configure with CMake (using vcpkg)
echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..\src\visualization\cpp
if errorlevel 1 (
    echo CMake configuration failed
    exit /b 1
)

:: Build
echo Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed
    exit /b 1
)

:: Copy the built module (handle version-specific naming)
echo Copying built module...
if exist "..\src\visualization\cpp\Release\viz_engine*.pyd" (
    for %%F in (..\src\visualization\cpp\Release\viz_engine*.pyd) do (
        copy "%%F" "..\src\visualization\cpp\viz_engine.pyd"
        echo Copied %%F to viz_engine.pyd
    )
) else (
    echo Build failed: viz_engine.pyd not found
    exit /b 1
)

cd ..
echo Done!
pause