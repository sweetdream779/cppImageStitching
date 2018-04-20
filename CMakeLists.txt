cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++11")

project(stitch)

set(EIGEN_INCLUDE_DIR "eigen" CACHE PATH "Where is the include directory of Eigen located")
set(OpenCV_DIR "/opt/opencv3/share/OpenCV")
find_package(OpenCV REQUIRED)

include_directories("./include" ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR} "poison_im_editing_inc")

add_subdirectory("./src")  

