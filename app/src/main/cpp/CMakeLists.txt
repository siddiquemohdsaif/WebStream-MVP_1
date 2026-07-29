cmake_minimum_required(VERSION 3.22.1)
project(camera_pipeline LANGUAGES C CXX)

find_package(libjpeg-turbo REQUIRED CONFIG)

add_library(camera_pipeline SHARED native-camera.cpp native-vulkan-renderer.cpp
        remote-yuv-vulkan-renderer.cpp
        native-420-vulkan-renderer.cpp
        native-vulkan-feature-test.cpp
        native-yuv-jpeg.cpp
        camera_preprocess_pipeline.cpp
        native-preprocess-downsample.cpp
        native-preprocess-sobel.cpp
        native-preprocess-median-v2.cpp
        native-preprocess-full-optimized.cpp
        native-preprocess-full-optimized-2.cpp
        camera_transformation_evaluator.c)
target_compile_features(camera_pipeline PRIVATE cxx_std_17)
target_compile_options(camera_pipeline PRIVATE -O3 -Wall -Wextra)
target_link_libraries(camera_pipeline PRIVATE vulkan log android jnigraphics camera2ndk mediandk libjpeg-turbo::libturbojpeg.so)

