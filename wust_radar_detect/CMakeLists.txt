cmake_minimum_required(VERSION 3.8)
project(wust_radar_detect)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
add_compile_options(-w)  # 关闭所有 warning

# 添加 CUDA 架构（根据你设备实际情况）
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# 设置模块路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

# 查找依赖包
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV      REQUIRED)
find_package(HikSDK      REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(wust_utils  REQUIRED)
find_package(Eigen3      REQUIRED)
find_package(rclcpp      REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(wust_radar_interfaces REQUIRED)
find_package(wust_vl REQUIRED)

# 设置 TensorRT 路径并查找库
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-10.6.0.26")
find_package(TensorRT REQUIRED)

# 查找 nvonnxparser 动态库
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /home/hy/TensorRT-10.6.0.26/lib)
if(NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR "libnvonnxparser.so not found")
endif()

ament_auto_find_build_dependencies()


add_subdirectory(infer)
include_directories(${PROJECT_SOURCE_DIR}/infer/include)

# 添加主库
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

# 头文件包含路径（采用 scoped include 规范）
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>          # include/wust_radar_detect 下的头文件
  $<INSTALL_INTERFACE:include>                              # 安装后走 install/include/wust_radar_detect
  ${PROJECT_SOURCE_DIR}/infer/include
  ${OpenCV_INCLUDE_DIRS}
  ${HikSDK_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIR}
  ${WUST_VL_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${HikSDK_LIB}
  ${WUST_VL_LIBS}
  TensorRT::TensorRT
  CUDA::cudart
  yaml-cpp
  /home/hy/TensorRT-10.6.0.26/lib/libnvonnxparser.so
  infer
)

# 定义宏
add_definitions(-D use_tensorrt)

# 注册 ROS2 组件
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN WustRadarDetectNode
  EXECUTABLE ${PROJECT_NAME}_node
)

# 安装资源
ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  model
  USE_SCOPED_HEADER_INSTALL_DIR 
)
