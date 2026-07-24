#include <jni.h>
#include <vulkan/vulkan.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>

#include "camera_preprocess_pipeline.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace {

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
    }
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct DownsamplePush {
    uint32_t src_w_u32;
    uint32_t dst_w_u32;
    uint32_t dst_base_word;
    uint32_t tile_count_x;
    uint32_t tile_count_y;
};

struct SobelPush {
    uint32_t width;
    uint32_t height;
    uint32_t strideWords;
    uint32_t planeWords;
};

struct MedianPush {
    uint32_t width;
    uint32_t height;
    uint32_t strideWords;
    uint32_t planeWords;
    uint32_t edgeThresholdY;
    uint32_t edgeThresholdChroma;
};

std::vector<uint8_t> bytesFromJava(JNIEnv* env, jbyteArray array) {
    const jsize size = env->GetArrayLength(array);
    std::vector<uint8_t> out(static_cast<size_t>(size));
    env->GetByteArrayRegion(array, 0, size, reinterpret_cast<jbyte*>(out.data()));
    return out;
}

uint8_t clampByte(int value) {
    return static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

class OptimizedPipeline {
public:
    OptimizedPipeline(const uint8_t* rgba,
                      uint32_t rgbaStride,
                      uint32_t width,
                      uint32_t height,
                      uint32_t mode,
                      const std::vector<uint8_t>& downsampleShader,
                      const std::vector<uint8_t>& sobelShader,
                      const std::vector<uint8_t>& medianShader,
                      uint32_t yThreshold,
                      uint32_t chromaThreshold)
        : rgba_(rgba),
          rgbaStride_(rgbaStride),
          srcWidth_(width),
          srcHeight_(height),
          mode_(mode),
          downsampleShader_(downsampleShader),
          sobelShader_(sobelShader),
          medianShader_(medianShader),
          yThreshold_(yThreshold),
          chromaThreshold_(chromaThreshold) {
        if (srcWidth_ == 0 || srcHeight_ == 0 || (srcWidth_ & 3u) != 0u) {
            throw std::runtime_error("Invalid source dimensions");
        }
        if (mode_ != 1u && mode_ != 3u && mode_ != 4u && mode_ != 5u) {
            throw std::runtime_error("Invalid downsample mode");
        }
        if (mode_ == 5u) {
            if (srcWidth_ % 16u != 0u || srcHeight_ % 4u != 0u) {
                throw std::runtime_error("4x4 -> 3x3 needs width multiple 16 and height multiple 4");
            }
            dstWidth_ = srcWidth_ / 4u * 3u;
            dstHeight_ = srcHeight_ / 4u * 3u;
            tileCountX_ = srcWidth_ / 16u;
            tileCountY_ = srcHeight_ / 4u;
        } else if (mode_ == 3u) {
            if (srcWidth_ % 12u != 0u || srcHeight_ % 3u != 0u) {
                throw std::runtime_error("3x3 -> 2x2 needs width multiple 12 and height multiple 3");
            }
            dstWidth_ = srcWidth_ / 3u * 2u;
            dstHeight_ = srcHeight_ / 3u * 2u;
            tileCountX_ = srcWidth_ / 12u;
            tileCountY_ = srcHeight_ / 3u;
        } else if (mode_ == 4u) {
            if (srcWidth_ % 8u != 0u || srcHeight_ % 2u != 0u) {
                throw std::runtime_error("4x4 -> 2x2 needs width multiple 8 and height multiple 2");
            }
            dstWidth_ = srcWidth_ / 2u;
            dstHeight_ = srcHeight_ / 2u;
            tileCountX_ = srcWidth_ / 8u;
            tileCountY_ = srcHeight_ / 2u;
        } else {
            dstWidth_ = srcWidth_;
            dstHeight_ = srcHeight_;
            tileCountX_ = srcWidth_ / 4u;
            tileCountY_ = srcHeight_;
        }
        srcStrideWords_ = srcWidth_ / 4u;
        dstStrideWords_ = dstWidth_ / 4u;
        srcPlaneWords_ = srcStrideWords_ * srcHeight_;
        dstPlaneWords_ = dstStrideWords_ * dstHeight_;

        createInstance();
        selectDevice();
        createDevice();
        createBuffers();
        createPipelines();
        createCommandObjects();
    }

    OptimizedPipeline(uint32_t width,
                      uint32_t height,
                      uint32_t mode,
                      const std::vector<uint8_t>& downsampleShader,
                      const std::vector<uint8_t>& sobelShader,
                      const std::vector<uint8_t>& medianShader,
                      uint32_t yThreshold,
                      uint32_t chromaThreshold)
        : OptimizedPipeline(nullptr, 0, width, height, mode,
                            downsampleShader, sobelShader, medianShader,
                            yThreshold, chromaThreshold) {}

    ~OptimizedPipeline() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            if (queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
            if (downsamplePipeline_) vkDestroyPipeline(device_, downsamplePipeline_, nullptr);
            if (sobelPipeline_) vkDestroyPipeline(device_, sobelPipeline_, nullptr);
            if (medianPipeline_) vkDestroyPipeline(device_, medianPipeline_, nullptr);
            if (downsampleLayout_) vkDestroyPipelineLayout(device_, downsampleLayout_, nullptr);
            if (sobelLayout_) vkDestroyPipelineLayout(device_, sobelLayout_, nullptr);
            if (medianLayout_) vkDestroyPipelineLayout(device_, medianLayout_, nullptr);
            if (downsampleSetLayout_) vkDestroyDescriptorSetLayout(device_, downsampleSetLayout_, nullptr);
            if (sobelSetLayout_) vkDestroyDescriptorSetLayout(device_, sobelSetLayout_, nullptr);
            if (medianSetLayout_) vkDestroyDescriptorSetLayout(device_, medianSetLayout_, nullptr);
            if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
            for (Buffer& buffer : sourceUploadPlanes_) destroy(buffer);
            for (Buffer& buffer : sourcePlanes_) destroy(buffer);
            destroy(compactUpload_);
            destroy(yuvDownsampled_);
            destroy(sobelMask_);
            destroy(yuvFiltered_);
            destroy(readback_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    double run() {
        const auto wallStart = std::chrono::high_resolution_clock::now();
        const auto uploadStart = std::chrono::high_resolution_clock::now();
        uploadRgbAsYuv444();
        const auto uploadEnd = std::chrono::high_resolution_clock::now();
        uploadMs_ = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
        cpuUploadMapCopyMs_ = uploadMs_;

        const auto recordStart = std::chrono::high_resolution_clock::now();
        check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer");
        const VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        check(vkBeginCommandBuffer(commandBuffer_, &begin), "vkBeginCommandBuffer");

        vkCmdResetQueryPool(commandBuffer_, queryPool_, 0, 10);
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 0);
        recordUploadCopy();
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool_, 1);
        recordUploadToShaderBarrier();
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 2);
        if (mode_ == 1u) {
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 3);
        } else {
            recordDownsample();
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 3);
            barrier(yuvDownsampled_.buffer, yuvDownsampled_.size,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 4);
        recordSobel();
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 5);
        barrier(sobelMask_.buffer, sobelMask_.size,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 6);
        recordMedian();
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_, 7);
        shaderToTransferBarrier(yuvFiltered_.buffer, yuvFiltered_.size);
        copyBuffer(yuvFiltered_, readback_);
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool_, 8);
        transferToHostBarrier(readback_.buffer, readback_.size);
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, 9);

        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        const auto recordEnd = std::chrono::high_resolution_clock::now();
        cpuCommandRecordMs_ = std::chrono::duration<double, std::milli>(recordEnd - recordStart).count();

        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
                                  0, nullptr, nullptr, 1, &commandBuffer_, 0, nullptr};
        const auto submitStart = std::chrono::high_resolution_clock::now();
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        const auto submitEnd = std::chrono::high_resolution_clock::now();
        cpuQueueSubmitMs_ = std::chrono::duration<double, std::milli>(submitEnd - submitStart).count();

        const auto waitStart = std::chrono::high_resolution_clock::now();
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE,
                              std::numeric_limits<uint64_t>::max()), "vkWaitForFences");
        const auto waitEnd = std::chrono::high_resolution_clock::now();
        cpuFenceWaitMs_ = std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();

        const auto resetFenceStart = std::chrono::high_resolution_clock::now();
        check(vkResetFences(device_, 1, &fence_), "vkResetFences");
        const auto resetFenceEnd = std::chrono::high_resolution_clock::now();
        cpuResetFenceMs_ = std::chrono::duration<double, std::milli>(resetFenceEnd - resetFenceStart).count();

        commandMs_ = cpuCommandRecordMs_ + cpuQueueSubmitMs_ + cpuFenceWaitMs_ + cpuResetFenceMs_;

        const auto queryStart = std::chrono::high_resolution_clock::now();
        std::array<uint64_t, 10> timestamps{};
        check(vkGetQueryPoolResults(device_, queryPool_, 0, 10,
                                    sizeof(timestamps), timestamps.data(),
                                    sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults");
        const auto queryEnd = std::chrono::high_resolution_clock::now();
        cpuQueryReadMs_ = std::chrono::duration<double, std::milli>(queryEnd - queryStart).count();

        const double periodMs = properties_.limits.timestampPeriod / 1'000'000.0;
        uploadCopyMs_ = static_cast<double>(timestamps[1] - timestamps[0]) * periodMs;
        uploadBarrierMs_ = static_cast<double>(timestamps[2] - timestamps[1]) * periodMs;
        downsampleMs_ = static_cast<double>(timestamps[3] - timestamps[2]) * periodMs;
        downsampleToSobelBarrierMs_ = static_cast<double>(timestamps[4] - timestamps[3]) * periodMs;
        sobelMs_ = static_cast<double>(timestamps[5] - timestamps[4]) * periodMs;
        sobelToMedianBarrierMs_ = static_cast<double>(timestamps[6] - timestamps[5]) * periodMs;
        medianMs_ = static_cast<double>(timestamps[7] - timestamps[6]) * periodMs;
        finalCopyMs_ = static_cast<double>(timestamps[8] - timestamps[7]) * periodMs;
        finalGpuToCpuBarrierMs_ = static_cast<double>(timestamps[9] - timestamps[8]) * periodMs;
        const auto wallEnd = std::chrono::high_resolution_clock::now();
        wallMs_ = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
        return commandMs_;
    }

    double runYuv444(const uint8_t* y, const uint8_t* u, const uint8_t* v) {
        yuvInput_[0] = y;
        yuvInput_[1] = u;
        yuvInput_[2] = v;
        return run();
    }

    void copyOutputTo(JNIEnv* env, jbyteArray outputArray) {
        const size_t bytes = static_cast<size_t>(dstPlaneWords_) * 3u * sizeof(uint32_t);
        if (env->GetArrayLength(outputArray) != static_cast<jsize>(bytes)) {
            throw std::runtime_error("Output array has wrong size");
        }
        void* mapped = nullptr;
        check(vkMapMemory(device_, readback_.memory, 0, readback_.size, 0, &mapped),
              "vkMapMemory(readback)");
        env->SetByteArrayRegion(outputArray, 0, static_cast<jsize>(bytes),
                                static_cast<const jbyte*>(mapped));
        vkUnmapMemory(device_, readback_.memory);
    }

    void copyOutputToRaw(uint8_t* outY,
                         uint8_t* outU,
                         uint8_t* outV,
                         size_t outCapacityPixels) {
        const auto outputStart = std::chrono::high_resolution_clock::now();
        const size_t outputPixels = static_cast<size_t>(dstWidth_) * dstHeight_;
        if (outputPixels > outCapacityPixels) {
            throw std::runtime_error("Output buffers are too small");
        }

        void* mapped = nullptr;
        check(vkMapMemory(device_, readback_.memory, 0, readback_.size, 0, &mapped),
              "vkMapMemory(readback)");
        const uint8_t* base = static_cast<const uint8_t*>(mapped);
        const size_t planeBytes = static_cast<size_t>(dstPlaneWords_) * sizeof(uint32_t);
        std::memcpy(outY, base, outputPixels);
        std::memcpy(outU, base + planeBytes, outputPixels);
        std::memcpy(outV, base + planeBytes * 2u, outputPixels);
        vkUnmapMemory(device_, readback_.memory);
        const auto outputEnd = std::chrono::high_resolution_clock::now();
        cpuOutputMapCopyMs_ = std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();
    }

    std::string report(double gpuMs) const {
        std::ostringstream out;
        out << properties_.deviceName << '\n'
            << srcWidth_ << 'x' << srcHeight_ << " -> "
            << dstWidth_ << 'x' << dstHeight_ << '\n'
            << "Single JNI call, single Vulkan instance/device\n"
            << "V2 memory layout: HOST_VISIBLE upload/readback, DEVICE_LOCAL intermediates\n"
            << "Intermediate buffers stay GPU/device-local:\n"
            << "RGBA -> YUV444 source planes -> downsampled YUV444 -> compact Sobel mask -> Median_v2 final\n"
            << "No Java byte[] round-trip between downsample, Sobel, and Median_v2\n"
            << "Mode: " << modeLabel() << '\n'
            << "Threshold Y=" << yThreshold_ << ", chroma=" << chromaThreshold_ << '\n'
            << std::fixed << std::setprecision(3)
            << "Timing:\n"
            << "  RGB -> YUV444 upload: " << uploadMs_ << " ms\n"
            << "  GPU upload copy to DEVICE_LOCAL: " << uploadCopyMs_ << " ms\n"
            << "  Barrier upload -> shader read: " << uploadBarrierMs_ << " ms\n"
            << "  Downsample: " << downsampleMs_ << " ms"
            << (mode_ == 1u ? " (skipped shader)" : "") << '\n'
            << "  Barrier downsample -> sobel: " << downsampleToSobelBarrierMs_ << " ms"
            << (mode_ == 1u ? " (transfer write -> shader read)" : " (shader write -> shader read)") << '\n'
            << "  Sobel mask: " << sobelMs_ << " ms\n"
            << "  Barrier sobel -> median_v2: " << sobelToMedianBarrierMs_ << " ms\n"
            << "  Median_v2: " << medianMs_ << " ms\n"
            << "  Final copy DEVICE_LOCAL -> readback: " << finalCopyMs_ << " ms\n"
            << "  Barrier readback -> CPU read: " << finalGpuToCpuBarrierMs_ << " ms\n"
            << "  Vulkan command submit+wait total: " << gpuMs << " ms";
        return out.str();
    }

    uint32_t dstWidth() const { return dstWidth_; }
    uint32_t dstHeight() const { return dstHeight_; }
    double uploadMs() const { return uploadMs_; }
    double uploadCopyMs() const { return uploadCopyMs_; }
    double uploadBarrierMs() const { return uploadBarrierMs_; }
    double downsampleMs() const { return downsampleMs_; }
    double downsampleToSobelBarrierMs() const { return downsampleToSobelBarrierMs_; }
    double sobelMs() const { return sobelMs_; }
    double sobelToMedianBarrierMs() const { return sobelToMedianBarrierMs_; }
    double medianMs() const { return medianMs_; }
    double finalCopyMs() const { return finalCopyMs_; }
    double finalGpuToCpuBarrierMs() const { return finalGpuToCpuBarrierMs_; }
    double commandMs() const { return commandMs_; }
    double cpuUploadMapCopyMs() const { return cpuUploadMapCopyMs_; }
    double cpuCommandRecordMs() const { return cpuCommandRecordMs_; }
    double cpuQueueSubmitMs() const { return cpuQueueSubmitMs_; }
    double cpuFenceWaitMs() const { return cpuFenceWaitMs_; }
    double cpuResetFenceMs() const { return cpuResetFenceMs_; }
    double cpuQueryReadMs() const { return cpuQueryReadMs_; }
    double cpuOutputMapCopyMs() const { return cpuOutputMapCopyMs_; }
    double wallMs() const { return wallMs_ + cpuOutputMapCopyMs_; }

private:
    const char* modeLabel() const {
        if (mode_ == 3u) return "3x3 -> 2x2";
        if (mode_ == 4u) return "4x4 -> 2x2";
        return "1x1 -> 1x1";
    }

    void createInstance() {
        const VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
                                        "OptimizedPreprocessing", 1,
                                        "OptimizedPreprocessing", 1, VK_API_VERSION_1_1};
        const VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
                                        &appInfo, 0, nullptr, 0, nullptr};
        check(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (count == 0) throw std::runtime_error("No Vulkan GPU found");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
        for (VkPhysicalDevice candidate : devices) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            for (uint32_t i = 0; i < familyCount; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physicalDevice_ = candidate;
                    queueFamily_ = i;
                    vkGetPhysicalDeviceProperties(candidate, &properties_);
                    vkGetPhysicalDeviceMemoryProperties(candidate, &memoryProperties_);
                    return;
                }
            }
        }
        throw std::runtime_error("No Vulkan compute queue found");
    }

    void createDevice() {
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
                                                queueFamily_, 1, &priority};
        const VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
                                      1, &queueInfo, 0, nullptr, 0, nullptr, nullptr};
        check(vkCreateDevice(physicalDevice_, &info, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    }

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags preferred) const {
        for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
            const auto flags = memoryProperties_.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) && (flags & preferred) == preferred) {
                return i;
            }
        }
        throw std::runtime_error("No compatible Vulkan memory type");
    }

    Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        Buffer out{};
        out.size = size;
        const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                                      size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        check(vkCreateBuffer(device_, &info, nullptr, &out.buffer), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &req);
        const VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                                         req.size, findMemoryType(req.memoryTypeBits, flags)};
        check(vkAllocateMemory(device_, &alloc, nullptr, &out.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");
        return out;
    }

    Buffer makeDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        return makeBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    Buffer makeHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        return makeBuffer(size, usage,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    void destroy(Buffer& buffer) {
        if (buffer.buffer) vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    }

    void createBuffers() {
        const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const VkBufferUsageFlags transferSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const VkBufferUsageFlags transferDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const VkDeviceSize sourceBytes = static_cast<VkDeviceSize>(srcPlaneWords_) * sizeof(uint32_t);
        for (uint32_t i = 0; i < 3u; ++i) {
            sourceUploadPlanes_[i] = makeHostBuffer(sourceBytes, transferSrc);
            sourcePlanes_[i] = makeDeviceBuffer(sourceBytes, storage | transferDst);
        }
        const VkDeviceSize compactBytes = static_cast<VkDeviceSize>(dstPlaneWords_) * 3u * sizeof(uint32_t);
        compactUpload_ = makeHostBuffer(compactBytes, transferSrc);
        yuvDownsampled_ = makeDeviceBuffer(compactBytes, storage | transferSrc | transferDst);
        sobelMask_ = makeDeviceBuffer(compactBytes, storage);
        yuvFiltered_ = makeDeviceBuffer(compactBytes, storage | transferSrc);
        readback_ = makeHostBuffer(compactBytes, transferDst);
    }

    VkShaderModule createShaderModule(const std::vector<uint8_t>& bytes) {
        if (bytes.empty() || (bytes.size() & 3u) != 0u) {
            throw std::runtime_error("Invalid SPIR-V asset");
        }
        const VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                            bytes.size(), reinterpret_cast<const uint32_t*>(bytes.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &info, nullptr, &module), "vkCreateShaderModule");
        return module;
    }

    VkPipeline createComputePipeline(const std::vector<uint8_t>& shader,
                                     VkPipelineLayout layout) {
        VkShaderModule module = createShaderModule(shader);
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                                                    module, "main", nullptr};
        const VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                               nullptr, 0, stage, layout, VK_NULL_HANDLE, -1};
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
        vkDestroyShaderModule(device_, module, nullptr);
        check(result, "vkCreateComputePipelines");
        return pipeline;
    }

    void createPipelines() {
        createDescriptorLayouts();
        createPipelineLayouts();
        downsamplePipeline_ = createComputePipeline(downsampleShader_, downsampleLayout_);
        sobelPipeline_ = createComputePipeline(sobelShader_, sobelLayout_);
        medianPipeline_ = createComputePipeline(medianShader_, medianLayout_);
        createDescriptorPoolAndSets();
    }

    void createDescriptorLayouts() {
        const VkDescriptorSetLayoutBinding downBindings[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const VkDescriptorSetLayoutCreateInfo downInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                       nullptr, 0, 2, downBindings};
        check(vkCreateDescriptorSetLayout(device_, &downInfo, nullptr, &downsampleSetLayout_),
              "vkCreateDescriptorSetLayout(downsample)");

        const VkDescriptorSetLayoutBinding sobelBindings[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const VkDescriptorSetLayoutCreateInfo sobelInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                        nullptr, 0, 2, sobelBindings};
        check(vkCreateDescriptorSetLayout(device_, &sobelInfo, nullptr, &sobelSetLayout_),
              "vkCreateDescriptorSetLayout(sobel)");

        const VkDescriptorSetLayoutBinding medianBindings[3] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const VkDescriptorSetLayoutCreateInfo medianInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                         nullptr, 0, 3, medianBindings};
        check(vkCreateDescriptorSetLayout(device_, &medianInfo, nullptr, &medianSetLayout_),
              "vkCreateDescriptorSetLayout(median)");
    }

    void createPipelineLayouts() {
        const VkPushConstantRange downRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DownsamplePush)};
        const VkPipelineLayoutCreateInfo downInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                  nullptr, 0, 1, &downsampleSetLayout_, 1, &downRange};
        check(vkCreatePipelineLayout(device_, &downInfo, nullptr, &downsampleLayout_),
              "vkCreatePipelineLayout(downsample)");

        const VkPushConstantRange sobelRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SobelPush)};
        const VkPipelineLayoutCreateInfo sobelInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                   nullptr, 0, 1, &sobelSetLayout_, 1, &sobelRange};
        check(vkCreatePipelineLayout(device_, &sobelInfo, nullptr, &sobelLayout_),
              "vkCreatePipelineLayout(sobel)");

        const VkPushConstantRange medianRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MedianPush)};
        const VkPipelineLayoutCreateInfo medianInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                    nullptr, 0, 1, &medianSetLayout_, 1, &medianRange};
        check(vkCreatePipelineLayout(device_, &medianInfo, nullptr, &medianLayout_),
              "vkCreatePipelineLayout(median)");
    }

    void createDescriptorPoolAndSets() {
        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12};
        const VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                  nullptr, 0, 5, 1, &poolSize};
        check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_),
              "vkCreateDescriptorPool");

        const VkDescriptorSetLayout downLayouts[3] = {
            downsampleSetLayout_, downsampleSetLayout_, downsampleSetLayout_};
        const VkDescriptorSetAllocateInfo downAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                    nullptr, descriptorPool_, 3, downLayouts};
        check(vkAllocateDescriptorSets(device_, &downAlloc, downsampleSets_.data()),
              "vkAllocateDescriptorSets(downsample)");

        const VkDescriptorSetAllocateInfo sobelAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                     nullptr, descriptorPool_, 1, &sobelSetLayout_};
        check(vkAllocateDescriptorSets(device_, &sobelAlloc, &sobelSet_),
              "vkAllocateDescriptorSets(sobel)");

        const VkDescriptorSetAllocateInfo medianAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                      nullptr, descriptorPool_, 1, &medianSetLayout_};
        check(vkAllocateDescriptorSets(device_, &medianAlloc, &medianSet_),
              "vkAllocateDescriptorSets(median)");

        updateDescriptorSets();
    }

    void updateDescriptorSets() {
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> infos;
        infos.reserve(12);

        auto addWrite = [&](VkDescriptorSet set, uint32_t binding, Buffer& buffer) {
            infos.push_back({buffer.buffer, 0, buffer.size});
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &infos.back();
            writes.push_back(write);
        };

        for (uint32_t c = 0; c < 3u; ++c) {
            addWrite(downsampleSets_[c], 0, sourcePlanes_[c]);
            addWrite(downsampleSets_[c], 1, yuvDownsampled_);
        }
        addWrite(sobelSet_, 0, yuvDownsampled_);
        addWrite(sobelSet_, 1, sobelMask_);
        addWrite(medianSet_, 0, yuvDownsampled_);
        addWrite(medianSet_, 1, sobelMask_);
        addWrite(medianSet_, 2, yuvFiltered_);

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void createCommandObjects() {
        const VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamily_};
        check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");
        const VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                      nullptr, commandPool_,
                                                      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        check(vkAllocateCommandBuffers(device_, &commandInfo, &commandBuffer_), "vkAllocateCommandBuffers");
        const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence");
        const VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                              VK_QUERY_TYPE_TIMESTAMP, 10, 0};
        check(vkCreateQueryPool(device_, &queryInfo, nullptr, &queryPool_), "vkCreateQueryPool");
    }

    void uploadRgbAsYuv444() {
        if (yuvInput_[0] && yuvInput_[1] && yuvInput_[2]) {
            uploadYuv444();
            return;
        }

        if (mode_ == 1u) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, compactUpload_.memory, 0, compactUpload_.size, 0, &mapped),
                  "vkMapMemory(compactUpload)");
            uint8_t* base = static_cast<uint8_t*>(mapped);
            uint8_t* yPlane = base;
            uint8_t* cbPlane = base + static_cast<size_t>(dstPlaneWords_) * sizeof(uint32_t);
            uint8_t* crPlane = cbPlane + static_cast<size_t>(dstPlaneWords_) * sizeof(uint32_t);
            for (uint32_t y = 0; y < srcHeight_; ++y) {
                const uint8_t* row = rgba_ + static_cast<size_t>(y) * rgbaStride_;
                for (uint32_t x = 0; x < srcWidth_; ++x) {
                    const uint8_t r = row[x * 4u + 0u];
                    const uint8_t g = row[x * 4u + 1u];
                    const uint8_t b = row[x * 4u + 2u];
                    const size_t i = static_cast<size_t>(y) * srcWidth_ + x;
                    yPlane[i] = clampByte((77 * r + 150 * g + 29 * b + 128) >> 8);
                    cbPlane[i] = clampByte(128 + ((-43 * r - 85 * g + 128 * b + 128) >> 8));
                    crPlane[i] = clampByte(128 + ((128 * r - 107 * g - 21 * b + 128) >> 8));
                }
            }
            vkUnmapMemory(device_, compactUpload_.memory);
            return;
        }

        std::array<uint8_t*, 3> planes{};
        for (uint32_t c = 0; c < 3u; ++c) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, sourceUploadPlanes_[c].memory, 0,
                              sourceUploadPlanes_[c].size, 0, &mapped),
                  "vkMapMemory(sourceUpload)");
            planes[c] = static_cast<uint8_t*>(mapped);
        }
        for (uint32_t y = 0; y < srcHeight_; ++y) {
            const uint8_t* row = rgba_ + static_cast<size_t>(y) * rgbaStride_;
            for (uint32_t x = 0; x < srcWidth_; ++x) {
                const uint8_t r = row[x * 4u + 0u];
                const uint8_t g = row[x * 4u + 1u];
                const uint8_t b = row[x * 4u + 2u];
                const size_t i = static_cast<size_t>(y) * srcWidth_ + x;
                planes[0][i] = clampByte((77 * r + 150 * g + 29 * b + 128) >> 8);
                planes[1][i] = clampByte(128 + ((-43 * r - 85 * g + 128 * b + 128) >> 8));
                planes[2][i] = clampByte(128 + ((128 * r - 107 * g - 21 * b + 128) >> 8));
            }
        }
        for (uint32_t c = 0; c < 3u; ++c) {
            vkUnmapMemory(device_, sourceUploadPlanes_[c].memory);
        }
    }

    void uploadYuv444() {
        if (mode_ == 1u) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, compactUpload_.memory, 0, compactUpload_.size, 0, &mapped),
                  "vkMapMemory(compactUpload)");
            uint8_t* base = static_cast<uint8_t*>(mapped);
            const size_t planeBytes = static_cast<size_t>(dstPlaneWords_) * sizeof(uint32_t);
            const size_t pixels = static_cast<size_t>(srcWidth_) * srcHeight_;
            std::memcpy(base, yuvInput_[0], pixels);
            std::memcpy(base + planeBytes, yuvInput_[1], pixels);
            std::memcpy(base + planeBytes * 2u, yuvInput_[2], pixels);
            vkUnmapMemory(device_, compactUpload_.memory);
            return;
        }

        for (uint32_t c = 0; c < 3u; ++c) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, sourceUploadPlanes_[c].memory, 0,
                              sourceUploadPlanes_[c].size, 0, &mapped),
                  "vkMapMemory(sourceUpload)");
            std::memcpy(mapped, yuvInput_[c], static_cast<size_t>(srcWidth_) * srcHeight_);
            vkUnmapMemory(device_, sourceUploadPlanes_[c].memory);
        }
    }

    void copyBuffer(Buffer& src, Buffer& dst) {
        const VkBufferCopy copy{0, 0, src.size < dst.size ? src.size : dst.size};
        vkCmdCopyBuffer(commandBuffer_, src.buffer, dst.buffer, 1, &copy);
    }

    void recordUploadCopy() {
        if (mode_ == 1u) {
            copyBuffer(compactUpload_, yuvDownsampled_);
            return;
        }
        for (uint32_t c = 0; c < 3u; ++c) {
            copyBuffer(sourceUploadPlanes_[c], sourcePlanes_[c]);
        }
    }

    void recordUploadToShaderBarrier() {
        if (mode_ == 1u) {
            transferToShaderBarrier(yuvDownsampled_.buffer, yuvDownsampled_.size);
            return;
        }
        std::array<VkBufferMemoryBarrier, 3> barriers{};
        for (uint32_t c = 0; c < 3u; ++c) {
            barriers[c] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                           sourcePlanes_[c].buffer, 0, sourcePlanes_[c].size};
        }
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, static_cast<uint32_t>(barriers.size()),
                             barriers.data(), 0, nullptr);
    }

    void recordDownsample() {
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, downsamplePipeline_);
        for (uint32_t c = 0; c < 3u; ++c) {
            vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    downsampleLayout_, 0, 1, &downsampleSets_[c], 0, nullptr);
            const DownsamplePush push{srcStrideWords_, dstStrideWords_, c * dstPlaneWords_,
                                      tileCountX_, tileCountY_};
            vkCmdPushConstants(commandBuffer_, downsampleLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(push), &push);
            vkCmdDispatch(commandBuffer_, (tileCountX_ + 127u) / 128u, tileCountY_, 1);
            if (c + 1u < 3u) {
                barrier(yuvDownsampled_.buffer, yuvDownsampled_.size,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            }
        }
    }

    void recordSobel() {
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, sobelPipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sobelLayout_, 0, 1, &sobelSet_, 0, nullptr);
        const SobelPush push{dstWidth_, dstHeight_, dstStrideWords_, dstPlaneWords_};
        vkCmdPushConstants(commandBuffer_, sobelLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer_, (dstStrideWords_ + 31u) / 32u, dstHeight_, 3);
    }

    void recordMedian() {
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, medianPipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                medianLayout_, 0, 1, &medianSet_, 0, nullptr);
        const MedianPush push{dstWidth_, dstHeight_, dstStrideWords_, dstPlaneWords_,
                              yThreshold_, chromaThreshold_};
        vkCmdPushConstants(commandBuffer_, medianLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer_, (dstStrideWords_ + 31u) / 32u, dstHeight_, 3);
    }

    void barrier(VkBuffer buffer, VkDeviceSize size, VkAccessFlags src, VkAccessFlags dst) {
        const VkBufferMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                  src, dst,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                  buffer, 0, size};
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &memoryBarrier, 0, nullptr);
    }

    void transferToShaderBarrier(VkBuffer buffer, VkDeviceSize size) {
        const VkBufferMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                  buffer, 0, size};
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &memoryBarrier, 0, nullptr);
    }

    void shaderToTransferBarrier(VkBuffer buffer, VkDeviceSize size) {
        const VkBufferMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                  buffer, 0, size};
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &memoryBarrier, 0, nullptr);
    }

    void transferToHostBarrier(VkBuffer buffer, VkDeviceSize size) {
        const VkBufferMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                  buffer, 0, size};
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &memoryBarrier, 0, nullptr);
    }

    const uint8_t* rgba_ = nullptr;
    std::array<const uint8_t*, 3> yuvInput_{};
    uint32_t rgbaStride_ = 0;
    uint32_t srcWidth_ = 0;
    uint32_t srcHeight_ = 0;
    uint32_t dstWidth_ = 0;
    uint32_t dstHeight_ = 0;
    uint32_t mode_ = 0;
    uint32_t srcStrideWords_ = 0;
    uint32_t dstStrideWords_ = 0;
    uint32_t srcPlaneWords_ = 0;
    uint32_t dstPlaneWords_ = 0;
    uint32_t tileCountX_ = 0;
    uint32_t tileCountY_ = 0;
    std::vector<uint8_t> downsampleShader_;
    std::vector<uint8_t> sobelShader_;
    std::vector<uint8_t> medianShader_;
    uint32_t yThreshold_ = 80;
    uint32_t chromaThreshold_ = 40;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    uint32_t queueFamily_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::array<Buffer, 3> sourceUploadPlanes_{};
    std::array<Buffer, 3> sourcePlanes_{};
    Buffer compactUpload_{};
    Buffer yuvDownsampled_{};
    Buffer sobelMask_{};
    Buffer yuvFiltered_{};
    Buffer readback_{};
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout downsampleSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sobelSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout medianSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout downsampleLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout sobelLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout medianLayout_ = VK_NULL_HANDLE;
    VkPipeline downsamplePipeline_ = VK_NULL_HANDLE;
    VkPipeline sobelPipeline_ = VK_NULL_HANDLE;
    VkPipeline medianPipeline_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 3> downsampleSets_{};
    VkDescriptorSet sobelSet_ = VK_NULL_HANDLE;
    VkDescriptorSet medianSet_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    double uploadMs_ = 0.0;
    double uploadCopyMs_ = 0.0;
    double uploadBarrierMs_ = 0.0;
    double downsampleMs_ = 0.0;
    double downsampleToSobelBarrierMs_ = 0.0;
    double sobelMs_ = 0.0;
    double sobelToMedianBarrierMs_ = 0.0;
    double medianMs_ = 0.0;
    double finalCopyMs_ = 0.0;
    double finalGpuToCpuBarrierMs_ = 0.0;
    double commandMs_ = 0.0;
    double cpuUploadMapCopyMs_ = 0.0;
    double cpuCommandRecordMs_ = 0.0;
    double cpuQueueSubmitMs_ = 0.0;
    double cpuFenceWaitMs_ = 0.0;
    double cpuResetFenceMs_ = 0.0;
    double cpuQueryReadMs_ = 0.0;
    double cpuOutputMapCopyMs_ = 0.0;
    double wallMs_ = 0.0;
};

} // namespace

namespace {

std::mutex gCameraGpuPipelineMutex;
std::unique_ptr<OptimizedPipeline> gCameraGpuPipeline;
std::array<std::vector<uint8_t>, 4> gCameraDownsampleShaders;
std::vector<uint8_t> gCameraSobelShader;
std::vector<uint8_t> gCameraMedianShader;
uint32_t gCameraPipelineWidth = 0;
uint32_t gCameraPipelineHeight = 0;
uint32_t gCameraPipelineMode = 0;

std::vector<uint8_t> readAssetBytes(AAssetManager* assets, const char* name) {
    if (!assets) throw std::runtime_error("AssetManager is null");
    AAsset* asset = AAssetManager_open(assets, name, AASSET_MODE_BUFFER);
    if (!asset) throw std::runtime_error(std::string("Missing asset: ") + name);
    const off_t size = AAsset_getLength(asset);
    if (size <= 0) {
        AAsset_close(asset);
        throw std::runtime_error(std::string("Empty asset: ") + name);
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const int read = AAsset_read(asset, bytes.data(), size);
    AAsset_close(asset);
    if (read != size) throw std::runtime_error(std::string("Failed to read asset: ") + name);
    return bytes;
}

size_t downsampleShaderIndex(uint32_t mode) {
    if (mode == 4u) return 0u;
    if (mode == 5u) return 1u;
    if (mode == 3u) return 2u;
    return 3u;
}

} // namespace

extern "C" bool cameraPreprocessGpuLoadShaders(AAssetManager* assets) {
    std::lock_guard<std::mutex> lock(gCameraGpuPipelineMutex);
    try {
        gCameraDownsampleShaders[0] = readAssetBytes(assets, "downsample_4to2.comp.spv");
        gCameraDownsampleShaders[1] = readAssetBytes(assets, "downsample_4to3.comp.spv");
        gCameraDownsampleShaders[2] = readAssetBytes(assets, "downsample_3to2.comp.spv");
        gCameraDownsampleShaders[3] = readAssetBytes(assets, "downsample_1to1.comp.spv");
        gCameraSobelShader = readAssetBytes(assets, "optimized_sobel_compact.comp.spv");
        gCameraMedianShader = readAssetBytes(assets, "optimized_median_v2_compact.comp.spv");
        return true;
    } catch (...) {
        gCameraDownsampleShaders = {};
        gCameraSobelShader.clear();
        gCameraMedianShader.clear();
        gCameraGpuPipeline.reset();
        gCameraPipelineWidth = 0;
        gCameraPipelineHeight = 0;
        gCameraPipelineMode = 0;
        return false;
    }
}

extern "C" void cameraPreprocessGpuDestroy() {
    std::lock_guard<std::mutex> lock(gCameraGpuPipelineMutex);
    gCameraGpuPipeline.reset();
    gCameraPipelineWidth = 0;
    gCameraPipelineHeight = 0;
    gCameraPipelineMode = 0;
}

extern "C" bool cameraPreprocessGpuYuv444ToBuffer(
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
        CameraPreprocessGpuTiming* timing) {
    if (!y || !u || !v || !outY || !outU || !outV || !outWidth || !outHeight ||
        width <= 0 || height <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(gCameraGpuPipelineMutex);
    try {
        const uint32_t mode = static_cast<uint32_t>(downsampleType);
        const size_t shaderIndex = downsampleShaderIndex(mode);
        if (gCameraDownsampleShaders[shaderIndex].empty() ||
            gCameraSobelShader.empty() || gCameraMedianShader.empty()) {
            return false;
        }

        const uint32_t w = static_cast<uint32_t>(width);
        const uint32_t h = static_cast<uint32_t>(height);
        if (!gCameraGpuPipeline ||
            gCameraPipelineWidth != w ||
            gCameraPipelineHeight != h ||
            gCameraPipelineMode != mode) {
            gCameraGpuPipeline = std::make_unique<OptimizedPipeline>(
                    w, h, mode,
                    gCameraDownsampleShaders[shaderIndex],
                    gCameraSobelShader,
                    gCameraMedianShader,
                    80u, 40u);
            gCameraPipelineWidth = w;
            gCameraPipelineHeight = h;
            gCameraPipelineMode = mode;
        }

        gCameraGpuPipeline->runYuv444(y, u, v);
        gCameraGpuPipeline->copyOutputToRaw(outY, outU, outV, outCapacityPixels);
        *outWidth = static_cast<int>(gCameraGpuPipeline->dstWidth());
        *outHeight = static_cast<int>(gCameraGpuPipeline->dstHeight());

        if (timing) {
            timing->uploadMs = gCameraGpuPipeline->uploadMs();
            timing->uploadCopyMs = gCameraGpuPipeline->uploadCopyMs();
            timing->uploadBarrierMs = gCameraGpuPipeline->uploadBarrierMs();
            timing->downsampleMs = gCameraGpuPipeline->downsampleMs();
            timing->downsampleToSobelBarrierMs = gCameraGpuPipeline->downsampleToSobelBarrierMs();
            timing->sobelMs = gCameraGpuPipeline->sobelMs();
            timing->sobelToMedianBarrierMs = gCameraGpuPipeline->sobelToMedianBarrierMs();
            timing->medianMs = gCameraGpuPipeline->medianMs();
            timing->finalCopyMs = gCameraGpuPipeline->finalCopyMs();
            timing->finalGpuToCpuBarrierMs = gCameraGpuPipeline->finalGpuToCpuBarrierMs();
            timing->commandMs = gCameraGpuPipeline->commandMs();
            timing->cpuUploadMapCopyMs = gCameraGpuPipeline->cpuUploadMapCopyMs();
            timing->cpuCommandRecordMs = gCameraGpuPipeline->cpuCommandRecordMs();
            timing->cpuQueueSubmitMs = gCameraGpuPipeline->cpuQueueSubmitMs();
            timing->cpuFenceWaitMs = gCameraGpuPipeline->cpuFenceWaitMs();
            timing->cpuResetFenceMs = gCameraGpuPipeline->cpuResetFenceMs();
            timing->cpuQueryReadMs = gCameraGpuPipeline->cpuQueryReadMs();
            timing->cpuOutputMapCopyMs = gCameraGpuPipeline->cpuOutputMapCopyMs();
            timing->wallMs = gCameraGpuPipeline->wallMs();
        }
        return true;
    } catch (...) {
        gCameraGpuPipeline.reset();
        gCameraPipelineWidth = 0;
        gCameraPipelineHeight = 0;
        gCameraPipelineMode = 0;
        return false;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_OptimizedPreprocessingTest_12_nativeRunFullPipeline(
    JNIEnv* env,
    jclass,
    jobject bitmap,
    jbyteArray outputArray,
    jint width,
    jint height,
    jint mode,
    jbyteArray downsampleShader,
    jbyteArray sobelShader,
    jbyteArray medianShader,
    jint yThreshold,
    jint chromaThreshold) {
    void* bitmapPixels = nullptr;
    try {
        AndroidBitmapInfo bitmapInfo{};
        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS ||
            bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
            static_cast<uint32_t>(width) > bitmapInfo.width ||
            static_cast<uint32_t>(height) > bitmapInfo.height) {
            throw std::runtime_error("Bitmap must be RGBA_8888 and cover trimmed dimensions");
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
            throw std::runtime_error("AndroidBitmap_lockPixels failed");
        }

        OptimizedPipeline pipeline(static_cast<const uint8_t*>(bitmapPixels),
                                   bitmapInfo.stride,
                                   static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height),
                                   static_cast<uint32_t>(mode),
                                   bytesFromJava(env, downsampleShader),
                                   bytesFromJava(env, sobelShader),
                                   bytesFromJava(env, medianShader),
                                   static_cast<uint32_t>(yThreshold),
                                   static_cast<uint32_t>(chromaThreshold));
        AndroidBitmap_unlockPixels(env, bitmap);
        bitmapPixels = nullptr;

        const double gpuMs = pipeline.run();
        pipeline.copyOutputTo(env, outputArray);
        return env->NewStringUTF(pipeline.report(gpuMs).c_str());
    } catch (const std::exception& error) {
        if (bitmapPixels) AndroidBitmap_unlockPixels(env, bitmap);
        return env->NewStringUTF((std::string("Error: ") + error.what()).c_str());
    }
}
