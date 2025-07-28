cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)

project(yolo LANGUAGES CXX CUDA)

# 基础编译设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 抑制一些过时 API 警告
add_compile_options(-Wno-deprecated-declarations)

# 查找 CUDA 和 OpenCV
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# 这里不调用 find_package(TensorRT)，统一由顶层管理

# 源码文件收集
file(GLOB_RECURSE YOLO_SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

# 添加静态库
add_library(yolo STATIC ${YOLO_SRC})
set_target_properties(yolo PROPERTIES POSITION_INDEPENDENT_CODE ON)
# 头文件目录（含 CUDA 和当前目录）
target_include_directories(yolo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR} 
)

# 链接库：OpenCV 和 CUDA，TensorRT 在顶层链接给主库
target_link_libraries(yolo
    ${OpenCV_LIBS}
    ${CUDAToolkit_LIBRARY}
    TensorRT::TensorRT
    CUDA::cudart
)
