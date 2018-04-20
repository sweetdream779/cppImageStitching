
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++11")

# Library

add_library(blend 
	../poison_im_editing_inc/blend/poisson_solver.h
	../poison_im_editing_inc/blend/clone.h
	../poison_im_editing_inc/blend/blend.h
	../poison_im_editing_src/poisson_solver.cpp
	../poison_im_editing_src/clone.cpp	
	../poison_im_editing_src/blend.cpp
)

target_link_libraries(blend ${OpenCV_LIBS})

set(SOURCE_FILES homographyManager.cpp myStitcher.cpp reconstructer.cpp graphCutsMinimizer.cpp GCoptimization.cpp LinkedBlockList.cpp)

add_executable(main main.cpp ${SOURCE_FILES})
target_link_libraries(main ${OpenCV_LIBS} blend)
