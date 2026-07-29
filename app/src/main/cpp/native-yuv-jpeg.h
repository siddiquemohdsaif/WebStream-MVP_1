#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" bool nativeYuv420ToJpeg(
        const uint8_t* y,
        const uint8_t* u,
        const uint8_t* v,
        int width,
        int height,
        int yStride,
        int uvStride,
        int quality,
        std::vector<uint8_t>& jpeg);

extern "C" bool nativeJpegToYuv420(
        const uint8_t* jpeg,
        size_t jpegSize,
        std::vector<uint8_t>& yuv420,
        int& width,
        int& height);
