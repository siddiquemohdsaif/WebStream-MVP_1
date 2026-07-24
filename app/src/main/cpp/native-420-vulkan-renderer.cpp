#include "native-vulkan-renderer.h"

// YUV420 preprocessing/rendering uses the shared Vulkan renderer state in
// native-vulkan-renderer.cpp because the swapchain, SurfaceView, command queue,
// descriptor pools and render textures must be owned by one renderer instance.
//
// Public API:
//   vulkanSubmitPreprocessedYuv420(...)
//
// Flow:
//   CPU clean YUV420 planes
//     -> upload packed YUV420
//     -> GPU downsample YUV420 -> YUV420
//     -> GPU Sobel/median per plane, preserving Y and UV dimensions
//     -> GPU YUV420 textures
//     -> camera.frag YUV->RGB
//     -> swapchain
