
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(OpenCV_DIR "/opt/opencv3/share/OpenCV")
find_package(OpenCV REQUIRED)

set(SOURCE_FILES homographyManager.cpp myStitcher.cpp reconstructer.cpp graphCutsMinimizer.cpp GCoptimization.cpp LinkedBlockList.cpp)

add_executable(main main.cpp ${SOURCE_FILES})
target_link_libraries(main ${OpenCV_LIBS})
