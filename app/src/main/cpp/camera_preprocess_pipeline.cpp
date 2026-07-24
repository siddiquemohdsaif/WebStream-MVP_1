#include "camera_preprocess_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

uint8_t clampByte(int value) {
    return static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

uint8_t median9(std::array<uint8_t, 9> values) {
    std::nth_element(values.begin(), values.begin() + 4, values.end());
    return values[4];
}

uint8_t sobelAt(const std::vector<uint8_t>& plane, int width, int height, int x, int y) {
    if (x <= 0 || y <= 0 || x + 1 >= width || y + 1 >= height) return 0;
    auto at = [&](int px, int py) -> int {
        return plane[static_cast<size_t>(py) * width + px];
    };
    const int tl = at(x - 1, y - 1);
    const int tc = at(x, y - 1);
    const int tr = at(x + 1, y - 1);
    const int ml = at(x - 1, y);
    const int mr = at(x + 1, y);
    const int bl = at(x - 1, y + 1);
    const int bc = at(x, y + 1);
    const int br = at(x + 1, y + 1);
    const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
    const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
    return clampByte(static_cast<int>(std::sqrt(static_cast<float>(gx * gx + gy * gy))));
}

void sobelPlane(const std::vector<uint8_t>& input, int width, int height, std::vector<uint8_t>& mask) {
    mask.assign(static_cast<size_t>(width) * height, 0);
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            mask[static_cast<size_t>(y) * width + x] = sobelAt(input, width, height, x, y);
        }
    }
}

void medianV2Plane(const std::vector<uint8_t>& input,
                   const std::vector<uint8_t>& mask,
                   int width,
                   int height,
                   int threshold,
                   std::vector<uint8_t>& output) {
    output = input;
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            if (mask[i] > threshold) {
                output[i] = input[i];
                continue;
            }
            std::array<uint8_t, 9> values{};
            int k = 0;
            for (int yy = -1; yy <= 1; ++yy) {
                for (int xx = -1; xx <= 1; ++xx) {
                    values[k++] = input[static_cast<size_t>(y + yy) * width + (x + xx)];
                }
            }
            output[i] = median9(values);
        }
    }
}

void copyPlane(const uint8_t* src, int width, int height, std::vector<uint8_t>& dst) {
    dst.assign(src, src + static_cast<size_t>(width) * height);
}

uint8_t srcAt(const uint8_t* src, int width, int height, int x, int y) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return src[static_cast<size_t>(y) * width + x];
}

void downsample3To2Plane(const uint8_t* src, int srcW, int srcH, std::vector<uint8_t>& dst) {
    const int dstW = srcW / 3 * 2;
    const int dstH = srcH / 3 * 2;
    dst.assign(static_cast<size_t>(dstW) * dstH, 0);
    for (int ty = 0; ty < dstH / 2; ++ty) {
        for (int tx = 0; tx < dstW / 2; ++tx) {
            const int sx = tx * 3;
            const int sy = ty * 3;
            const int A = srcAt(src, srcW, srcH, sx + 0, sy + 0);
            const int B = srcAt(src, srcW, srcH, sx + 1, sy + 0);
            const int C = srcAt(src, srcW, srcH, sx + 2, sy + 0);
            const int D = srcAt(src, srcW, srcH, sx + 0, sy + 1);
            const int E = srcAt(src, srcW, srcH, sx + 1, sy + 1);
            const int F = srcAt(src, srcW, srcH, sx + 2, sy + 1);
            const int G = srcAt(src, srcW, srcH, sx + 0, sy + 2);
            const int H = srcAt(src, srcW, srcH, sx + 1, sy + 2);
            const int I = srcAt(src, srcW, srcH, sx + 2, sy + 2);
            auto div9 = [](int value) { return static_cast<uint8_t>((value + 4) / 9); };
            dst[static_cast<size_t>(ty * 2 + 0) * dstW + (tx * 2 + 0)] = div9(4 * A + 2 * B + 2 * D + E);
            dst[static_cast<size_t>(ty * 2 + 0) * dstW + (tx * 2 + 1)] = div9(2 * B + 4 * C + E + 2 * F);
            dst[static_cast<size_t>(ty * 2 + 1) * dstW + (tx * 2 + 0)] = div9(2 * D + E + 4 * G + 2 * H);
            dst[static_cast<size_t>(ty * 2 + 1) * dstW + (tx * 2 + 1)] = div9(E + 2 * F + 2 * H + 4 * I);
        }
    }
}

void downsample4To2Plane(const uint8_t* src, int srcW, int srcH, std::vector<uint8_t>& dst) {
    const int dstW = srcW / 2;
    const int dstH = srcH / 2;
    dst.assign(static_cast<size_t>(dstW) * dstH, 0);
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            const int sx = x * 2;
            const int sy = y * 2;
            const int sum = srcAt(src, srcW, srcH, sx, sy) +
                            srcAt(src, srcW, srcH, sx + 1, sy) +
                            srcAt(src, srcW, srcH, sx, sy + 1) +
                            srcAt(src, srcW, srcH, sx + 1, sy + 1);
            dst[static_cast<size_t>(y) * dstW + x] = static_cast<uint8_t>((sum + 2) / 4);
        }
    }
}

void downsample4To3Plane(const uint8_t* src, int srcW, int srcH, std::vector<uint8_t>& dst) {
    const int dstW = srcW / 4 * 3;
    const int dstH = srcH / 4 * 3;
    dst.assign(static_cast<size_t>(dstW) * dstH, 0);
    const int weights[3][4] = {
            {3, 1, 0, 0},
            {0, 2, 2, 0},
            {0, 0, 1, 3},
    };
    for (int ty = 0; ty < dstH / 3; ++ty) {
        for (int tx = 0; tx < dstW / 3; ++tx) {
            const int sx = tx * 4;
            const int sy = ty * 4;
            for (int oy = 0; oy < 3; ++oy) {
                for (int ox = 0; ox < 3; ++ox) {
                    int sum = 0;
                    for (int iy = 0; iy < 4; ++iy) {
                        for (int ix = 0; ix < 4; ++ix) {
                            sum += srcAt(src, srcW, srcH, sx + ix, sy + iy) *
                                   weights[oy][iy] * weights[ox][ix];
                        }
                    }
                    dst[static_cast<size_t>(ty * 3 + oy) * dstW + (tx * 3 + ox)] =
                            static_cast<uint8_t>((sum + 8) / 16);
                }
            }
        }
    }
}

bool downsamplePlane(const uint8_t* src,
                     int srcW,
                     int srcH,
                     int downsampleType,
                     std::vector<uint8_t>& dst,
                     int& dstW,
                     int& dstH) {
    if (downsampleType == CAMERA_DOWNSAMPLE_1X1_TO_1X1) {
        dstW = srcW;
        dstH = srcH;
        copyPlane(src, srcW, srcH, dst);
        return true;
    }
    if (downsampleType == CAMERA_DOWNSAMPLE_3X3_TO_2X2) {
        dstW = srcW / 3 * 2;
        dstH = srcH / 3 * 2;
        if (dstW <= 0 || dstH <= 0) return false;
        downsample3To2Plane(src, srcW, srcH, dst);
        return true;
    }
    if (downsampleType == CAMERA_DOWNSAMPLE_4X4_TO_2X2) {
        dstW = srcW / 2;
        dstH = srcH / 2;
        if (dstW <= 0 || dstH <= 0) return false;
        downsample4To2Plane(src, srcW, srcH, dst);
        return true;
    }
    if (downsampleType == CAMERA_DOWNSAMPLE_4X4_TO_3X3) {
        dstW = srcW / 4 * 3;
        dstH = srcH / 4 * 3;
        if (dstW <= 0 || dstH <= 0) return false;
        downsample4To3Plane(src, srcW, srcH, dst);
        return true;
    }
    return false;
}

} // namespace

bool cameraPreprocessYuv444(
        const uint8_t* y,
        const uint8_t* u,
        const uint8_t* v,
        int width,
        int height,
        int downsampleType,
        CameraYuv444PipelineOutput& output) {
    if (!y || !u || !v || width <= 0 || height <= 0) return false;

    std::vector<uint8_t> dy, du, dv;
    int outW = 0;
    int outH = 0;
    if (!downsamplePlane(y, width, height, downsampleType, dy, outW, outH)) return false;

    int chromaOutW = 0;
    int chromaOutH = 0;
    if (!downsamplePlane(u, width, height, downsampleType, du, chromaOutW, chromaOutH)) return false;
    if (chromaOutW != outW || chromaOutH != outH) return false;
    if (!downsamplePlane(v, width, height, downsampleType, dv, chromaOutW, chromaOutH)) return false;
    if (chromaOutW != outW || chromaOutH != outH) return false;

    std::vector<uint8_t> maskY, maskU, maskV;
    sobelPlane(dy, outW, outH, maskY);
    sobelPlane(du, outW, outH, maskU);
    sobelPlane(dv, outW, outH, maskV);

    output.width = outW;
    output.height = outH;
    medianV2Plane(dy, maskY, outW, outH, 80, output.y);
    medianV2Plane(du, maskU, outW, outH, 40, output.u);
    medianV2Plane(dv, maskV, outW, outH, 40, output.v);
    return true;
}

extern "C" bool cameraPreprocessYuv444ToBuffer(
        const uint8_t* y,
        const uint8_t* u,
        const uint8_t* v,
        int width,
        int height,
        int downsampleType,
        uint8_t* outY,
        uint8_t* outU,
        uint8_t* outV,
        size_t outCapacityPixels,
        int* outWidth,
        int* outHeight) {
    if (!outY || !outU || !outV || !outWidth || !outHeight) return false;

    CameraYuv444PipelineOutput output;
    if (!cameraPreprocessYuv444(y, u, v, width, height, downsampleType, output)) {
        return false;
    }

    const size_t outputPixels = static_cast<size_t>(output.width) * output.height;
    if (outputPixels > outCapacityPixels) return false;

    std::memcpy(outY, output.y.data(), outputPixels);
    std::memcpy(outU, output.u.data(), outputPixels);
    std::memcpy(outV, output.v.data(), outputPixels);
    *outWidth = output.width;
    *outHeight = output.height;
    return true;
}
