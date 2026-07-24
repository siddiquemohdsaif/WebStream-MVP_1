#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct AAssetManager;

#ifdef __cplusplus
#include <vector>
#endif

enum CameraDownsampleType {
    CAMERA_DOWNSAMPLE_1X1_TO_1X1 = 1,
    CAMERA_DOWNSAMPLE_3X3_TO_2X2 = 3,
    CAMERA_DOWNSAMPLE_4X4_TO_2X2 = 4,
    CAMERA_DOWNSAMPLE_4X4_TO_3X3 = 5,
};

#ifdef __cplusplus
extern "C" {
#endif

bool cameraPreprocessYuv444ToBuffer(
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
        int* outHeight);

typedef struct CameraPreprocessGpuTiming {
    double uploadMs;
    double uploadCopyMs;
    double uploadBarrierMs;
    double downsampleMs;
    double downsampleToSobelBarrierMs;
    double sobelMs;
    double sobelToMedianBarrierMs;
    double medianMs;
    double finalCopyMs;
    double finalGpuToCpuBarrierMs;
    double commandMs;
    double cpuUploadMapCopyMs;
    double cpuCommandRecordMs;
    double cpuQueueSubmitMs;
    double cpuFenceWaitMs;
    double cpuResetFenceMs;
    double cpuQueryReadMs;
    double cpuOutputMapCopyMs;
    double wallMs;
} CameraPreprocessGpuTiming;

bool cameraPreprocessGpuLoadShaders(struct AAssetManager* assets);
void cameraPreprocessGpuDestroy(void);
bool cameraPreprocessGpuYuv444ToBuffer(
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
        int* outHeight,
        CameraPreprocessGpuTiming* timing);

#ifdef __cplusplus
}

struct CameraYuv444PipelineOutput {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> y;
    std::vector<uint8_t> u;
    std::vector<uint8_t> v;
};

bool cameraPreprocessYuv444(
        const uint8_t* y,
        const uint8_t* u,
        const uint8_t* v,
        int width,
        int height,
        int downsampleType,
        CameraYuv444PipelineOutput& output);
#endif
