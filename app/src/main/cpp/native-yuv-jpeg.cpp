#include "native-yuv-jpeg.h"

#include <algorithm>
#include <turbojpeg.h>

extern "C" bool nativeYuv420ToJpeg(
        const uint8_t* y,
        const uint8_t* u,
        const uint8_t* v,
        int width,
        int height,
        int yStride,
        int uvStride,
        int quality,
        std::vector<uint8_t>& jpeg) {
    if (!y || !u || !v || width <= 0 || height <= 0 ||
        yStride < width || uvStride < (width + 1) / 2) {
        return false;
    }

    tjhandle handle = tjInitCompress();
    if (!handle) return false;

    const unsigned char* planes[3] = {y, u, v};
    const int strides[3] = {yStride, uvStride, uvStride};
    unsigned char* jpegBuffer = nullptr;
    unsigned long jpegSize = 0;
    const int jpegQuality = std::max(1, std::min(100, quality));
    const int result = tjCompressFromYUVPlanes(
            handle,
            planes,
            width,
            strides,
            height,
            TJSAMP_420,
            &jpegBuffer,
            &jpegSize,
            jpegQuality,
            TJFLAG_FASTDCT);

    if (result == 0 && jpegBuffer && jpegSize > 0) {
        jpeg.assign(jpegBuffer, jpegBuffer + jpegSize);
    }

    if (jpegBuffer) tjFree(jpegBuffer);
    tjDestroy(handle);
    return result == 0 && !jpeg.empty();
}

extern "C" bool nativeJpegToYuv420(
        const uint8_t* jpeg,
        size_t jpegSize,
        std::vector<uint8_t>& y,
        std::vector<uint8_t>& u,
        std::vector<uint8_t>& v,
        int& width,
        int& height) {
    width = 0;
    height = 0;
    y.clear();
    u.clear();
    v.clear();
    if (!jpeg || jpegSize == 0) return false;

    tjhandle handle = tjInitDecompress();
    if (!handle) return false;

    int jpegSubsamp = -1;
    int jpegColorspace = -1;
    int decodedWidth = 0;
    int decodedHeight = 0;
    bool success = false;
    if (tjDecompressHeader3(
                handle,
                jpeg,
                static_cast<unsigned long>(jpegSize),
                &decodedWidth,
                &decodedHeight,
                &jpegSubsamp,
                &jpegColorspace) == 0 &&
        decodedWidth > 1 && decodedHeight > 1 && jpegSubsamp == TJSAMP_420) {
        width = decodedWidth & ~1;
        height = decodedHeight & ~1;
        const int chromaWidth = width / 2;
        const int chromaHeight = height / 2;
        y.resize(static_cast<size_t>(width) * height);
        u.resize(static_cast<size_t>(chromaWidth) * chromaHeight);
        v.resize(static_cast<size_t>(chromaWidth) * chromaHeight);

        unsigned char* planes[3] = {y.data(), u.data(), v.data()};
        int strides[3] = {width, chromaWidth, chromaWidth};
        success = tjDecompressToYUVPlanes(
                handle,
                jpeg,
                static_cast<unsigned long>(jpegSize),
                planes,
                width,
                strides,
                height,
                TJFLAG_FASTDCT) == 0;
    }

    tjDestroy(handle);
    if (!success) {
        width = 0;
        height = 0;
        y.clear();
        u.clear();
        v.clear();
    }
    return success;
}
