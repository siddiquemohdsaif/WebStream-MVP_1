#pragma once
#include <android/asset_manager.h>
#include <android/native_window.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

std::string vulkanSetWindow(ANativeWindow* window, AAssetManager* assets);
void vulkanDestroy();
void vulkanSetPreviewRenderingEnabled(bool enabled);
bool vulkanConsumeLatestFilteredFramePresent();
void vulkanSubmitYuv420(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        int width, int height, uint16_t rotation, bool mirror);
void vulkanSubmitYuv444(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        int width, int height, uint16_t rotation, bool mirror);
void vulkanSubmitPreprocessedYuv444(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                                    int width, int height, int desiredWidth, int desiredHeight,
                                    uint16_t rotation, bool mirror);
void vulkanSubmitPreprocessedYuv420(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                                    int width, int height, int desiredWidth, int desiredHeight,
                                    uint16_t rotation, bool mirror);
bool vulkanCopyLatestFilteredArgb(std::vector<uint32_t>& argb, int& width, int& height);
bool vulkanCopyLatestFilteredYuv420(std::vector<uint8_t>& yuv,
                                    int& width,
                                    int& height,
                                    int& yStride,
                                    int& uvStride,
                                    int& yPlaneBytes,
                                    int& uvPlaneBytes);
bool vulkanCopyYuv420Rings(std::array<std::vector<uint8_t>, 3>& downsampledFrames,
                           std::array<std::vector<uint8_t>, 3>& filteredFrames,
                           int& width,
                           int& height,
                           int& chromaWidth,
                           int& chromaHeight,
                           int& paddedWidth,
                           int& paddedHeight,
                           int& paddedChromaWidth,
                           int& paddedChromaHeight);
