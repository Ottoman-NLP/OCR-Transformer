cmake_minimum_required(VERSION 3.10)
project(viz_engine)
include_directories(${VCPKG_INSTALLED_DIR}/x64-windows/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Find packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# Add ImGui implementation files
set(IMGUI_IMPL_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui_impl/imgui_impl_glfw.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui_impl/imgui_impl_opengl3.cpp"
)

# Create Python module
pybind11_add_module(viz_engine 
    viz_engine.cpp
    ${IMGUI_IMPL_FILES}
)

# Include directories (prioritize local headers)
target_include_directories(viz_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_impl
    ${Python_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    "C:/vcpkg/installed/x64-windows/include"
)

# Link libraries
target_link_libraries(viz_engine PRIVATE
    glfw
    imgui::imgui
    OpenGL::GL
)

# Set output directory
set_target_properties(viz_engine PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)