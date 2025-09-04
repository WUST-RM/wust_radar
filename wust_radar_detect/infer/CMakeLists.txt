cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)

project(infer LANGUAGES CXX CUDA)

# 基础编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
# 根据 TensorRT 编译环境选择，通常是 1
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

option(CUDA_USE_STATIC_CUDA_RUNTIME "Use static CUDA runtime" ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# CUDA & TensorRT
find_package(CUDAToolkit REQUIRED)
include(/home/hy/wust_radar/src/wust_radar_detect/cmake_modules/libnvinferConfig.cmake)

# include 目录
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${nvinfer_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

# OpenCV
find_package(OpenCV REQUIRED)

# 源文件
file(GLOB_RECURSE SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/BaseInfer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/InferTool.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NvidiaInterface.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/classify.cu
)

# 生成共享库 (SHARED)
add_library(infer SHARED ${SRCS})

# 链接库
target_link_libraries(infer
  ${OpenCV_LIBS}
  ${nvinfer_LIBRARY}
  ${nvonnxparser_LIBRARY}
  cudart
)
