#pragma once

#include <android/asset_manager.h>
#include <android/native_window.h>
#include <cstdint>
#include <cstddef>
#include <string>

std::string remoteVulkanSetWindow(ANativeWindow* window, AAssetManager* assets);
void remoteVulkanDestroy();
bool remoteVulkanRenderYuv420(const uint8_t* yuv420,
                              size_t yuv420Size,
                              int width,
                              int height,
                              uint16_t rotation,
                              bool mirror);
