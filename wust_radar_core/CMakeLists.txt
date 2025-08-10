cmake_minimum_required(VERSION 3.8)
project(wust_radar_core)
cmake_policy(SET CMP0072 NEW)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(wust_utils  REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(wust_radar_interfaces REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
ament_auto_find_build_dependencies()
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)
target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${Open3D_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  yaml-cpp
  Open3D::Open3D
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN WustRadarCoreNode
  EXECUTABLE ${PROJECT_NAME}_node
)

# 安装资源
ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)
