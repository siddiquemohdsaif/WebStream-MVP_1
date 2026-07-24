#include <jni.h>
#include <android/bitmap.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(result));
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct PushConstants {
    uint32_t srcWidthWords;
    uint32_t dstWidthWords;
    uint32_t dstBaseWord;
    uint32_t tileCountX;
    uint32_t tileCountY;
};

std::vector<uint8_t> bytesFromJava(JNIEnv* env, jbyteArray array) {
    std::vector<uint8_t> result(static_cast<size_t>(env->GetArrayLength(array)));
    env->GetByteArrayRegion(array, 0, static_cast<jsize>(result.size()),
                            reinterpret_cast<jbyte*>(result.data()));
    return result;
}

class VulkanDownsample {
public:
    VulkanDownsample(const uint8_t* rgba, uint32_t rgbaStride,
                     uint32_t width, uint32_t height,
                     uint32_t mode, const std::vector<uint8_t>& shader)
        : width_(width), height_(height), mode_(mode) {
        tileSrcW_ = mode == 5 ? 16u : (mode == 3 ? 12u : (mode == 4 ? 8u : 4u));
        tileSrcH_ = mode == 5 ? 4u : (mode == 3 ? 3u : (mode == 4 ? 2u : 1u));
        tileDstW_ = mode == 5 ? 12u : (mode == 3 ? 8u : 4u);
        tileDstH_ = mode == 5 ? 3u : (mode == 3 ? 2u : 1u);
        dstWidth_ = width_ / tileSrcW_ * tileDstW_;
        dstHeight_ = height_ / tileSrcH_ * tileDstH_;
        srcPlaneBytes_ = static_cast<VkDeviceSize>(width_) * height_;
        dstPlaneBytes_ = static_cast<VkDeviceSize>(dstWidth_) * dstHeight_;
        if (!rgba || width_ % 4u || dstWidth_ % 4u)
            throw std::runtime_error("Invalid planar YUV444 dimensions");
        createInstance();
        selectDevice();
        createDevice();
        createBuffers(rgba, rgbaStride);
        createDescriptorsAndPipeline(shader);
        createCommands();
    }

    ~VulkanDownsample() {
        if (device_) {
            vkDeviceWaitIdle(device_);
            if (queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
            if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            if (descriptorLayout_) vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
            destroy(deviceOutput_);
            for (auto& value : deviceInput_) destroy(value);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    double execute(uint32_t iterations, bool timed, bool copyOutput) {
        (void)copyOutput;
        begin();
        if (timed) {
            vkCmdResetQueryPool(commandBuffer_, queryPool_, 0, 2);
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 0);
        }
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        for (uint32_t run = 0; run < iterations; ++run) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipelineLayout_, 0, 1, &descriptorSets_[channel], 0, nullptr);
                const PushConstants pc{width_ / 4u, dstWidth_ / 4u,
                                       static_cast<uint32_t>(dstPlaneBytes_ / 4u) * channel,
                                       width_ / tileSrcW_, height_ / tileSrcH_};
                vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc), &pc);
                vkCmdDispatch(commandBuffer_, (pc.tileCountX + 127u) / 128u, pc.tileCountY, 1);
            }
            if (run + 1u < iterations) {
                const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                              VK_ACCESS_SHADER_WRITE_BIT,
                                              VK_ACCESS_SHADER_WRITE_BIT};
                vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &barrier, 0, nullptr, 0, nullptr);
            }
        }
        if (timed)
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, 1);
        submit();
        if (!timed) return 0.0;
        uint64_t ts[2]{};
        check(vkGetQueryPoolResults(device_, queryPool_, 0, 2, sizeof(ts), ts,
                                    sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults");
        return static_cast<double>(ts[1] - ts[0]) * properties_.limits.timestampPeriod / 1e6;
    }

    std::vector<uint8_t> output() {
        std::vector<uint8_t> result(static_cast<size_t>(dstPlaneBytes_ * 3u));
        void* mapped = nullptr;
        check(vkMapMemory(device_, deviceOutput_.memory, 0, deviceOutput_.size, 0, &mapped),
              "vkMapMemory(output)");
        std::memcpy(result.data(), mapped, result.size());
        vkUnmapMemory(device_, deviceOutput_.memory);
        return result;
    }

    const char* deviceName() const { return properties_.deviceName; }
    uint32_t dstWidth() const { return dstWidth_; }
    uint32_t dstHeight() const { return dstHeight_; }

private:
    void createInstance() {
        const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
                                    "SobotestDownsample", 1, "Sobotest", 1, VK_API_VERSION_1_1};
        const VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
                                        &app, 0, nullptr, 0, nullptr};
        check(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
        for (auto candidate : devices) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            for (uint32_t i = 0; i < familyCount; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[i].timestampValidBits) {
                    physicalDevice_ = candidate; queueFamily_ = i;
                    vkGetPhysicalDeviceProperties(candidate, &properties_);
                    vkGetPhysicalDeviceMemoryProperties(candidate, &memoryProperties_);
                    return;
                }
            }
        }
        throw std::runtime_error("No timestamp-capable Vulkan compute queue");
    }

    void createDevice() {
        float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
                                            queueFamily_, 1, &priority};
        const VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
                                      1, &queue, 0, nullptr, 0, nullptr, nullptr};
        check(vkCreateDevice(physicalDevice_, &info, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    }

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i)
            if ((bits & (1u << i)) &&
                (memoryProperties_.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("No compatible Vulkan memory type");
    }

    Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        Buffer out{}; out.size = size;
        const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                                      size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        check(vkCreateBuffer(device_, &info, nullptr, &out.handle), "vkCreateBuffer");
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device_, out.handle, &req);
        const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                                              req.size, memoryType(req.memoryTypeBits, flags)};
        check(vkAllocateMemory(device_, &allocation, nullptr, &out.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device_, out.handle, out.memory, 0), "vkBindBufferMemory");
        return out;
    }

    void destroy(Buffer& value) {
        if (value.handle) vkDestroyBuffer(device_, value.handle, nullptr);
        if (value.memory) vkFreeMemory(device_, value.memory, nullptr);
        value = {};
    }

    static uint8_t clampByte(int value) {
        return static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }

    void createBuffers(const uint8_t* rgba, uint32_t rgbaStride) {
        const auto zeroCopy = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::array<uint8_t*, 3> planes{};
        for (uint32_t channel = 0; channel < 3; ++channel) {
            deviceInput_[channel] = makeBuffer(srcPlaneBytes_,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               zeroCopy);
            void* mapped = nullptr;
            check(vkMapMemory(device_, deviceInput_[channel].memory, 0,
                              deviceInput_[channel].size, 0, &mapped),
                  "vkMapMemory(input)");
            planes[channel] = static_cast<uint8_t*>(mapped);
        }
        for (uint32_t y = 0; y < height_; ++y) {
            const uint8_t* row = rgba + static_cast<size_t>(y) * rgbaStride;
            for (uint32_t x = 0; x < width_; ++x) {
                const uint8_t r = row[x * 4u + 0u];
                const uint8_t g = row[x * 4u + 1u];
                const uint8_t b = row[x * 4u + 2u];
                const size_t i = static_cast<size_t>(y) * width_ + x;
                planes[0][i] = clampByte((77 * r + 150 * g + 29 * b + 128) >> 8);
                planes[1][i] = clampByte(128 + ((-43 * r - 85 * g + 128 * b + 128) >> 8));
                planes[2][i] = clampByte(128 + ((128 * r - 107 * g - 21 * b + 128) >> 8));
            }
        }
        for (uint32_t channel = 0; channel < 3; ++channel)
            vkUnmapMemory(device_, deviceInput_[channel].memory);
        deviceOutput_ = makeBuffer(dstPlaneBytes_ * 3u,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   zeroCopy);
    }

    void createDescriptorsAndPipeline(const std::vector<uint8_t>& shaderBytes) {
        const VkDescriptorSetLayoutBinding bindings[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                     nullptr, 0, 2, bindings};
        check(vkCreateDescriptorSetLayout(device_, &layout, nullptr, &descriptorLayout_),
              "vkCreateDescriptorSetLayout");
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        const VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                        nullptr, 0, 1, &descriptorLayout_, 1, &range};
        check(vkCreatePipelineLayout(device_, &pipelineLayout, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout");
        const VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                  shaderBytes.size(),
                                                  reinterpret_cast<const uint32_t*>(shaderBytes.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = module; stage.pName = "main";
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = stage; pipeline.layout = pipelineLayout_;
        check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_),
              "vkCreateComputePipelines");
        vkDestroyShaderModule(device_, module, nullptr);

        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
        const VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              nullptr, 0, 3, 1, &poolSize};
        check(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptorPool_), "vkCreateDescriptorPool");
        const VkDescriptorSetLayout layouts[3] = {descriptorLayout_, descriptorLayout_, descriptorLayout_};
        const VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   nullptr, descriptorPool_, 3, layouts};
        check(vkAllocateDescriptorSets(device_, &allocate, descriptorSets_.data()),
              "vkAllocateDescriptorSets");
        for (uint32_t i = 0; i < 3; ++i) {
            const VkDescriptorBufferInfo infos[2] = {
                {deviceInput_[i].handle, 0, deviceInput_[i].size},
                {deviceOutput_.handle, 0, deviceOutput_.size}};
            VkWriteDescriptorSet writes[2]{};
            for (uint32_t j = 0; j < 2; ++j) {
                writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[j].dstSet = descriptorSets_[i]; writes[j].dstBinding = j;
                writes[j].descriptorCount = 1; writes[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[j].pBufferInfo = &infos[j];
            }
            vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        }
    }

    void createCommands() {
        const VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                           VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamily_};
        check(vkCreateCommandPool(device_, &pool, nullptr, &commandPool_), "vkCreateCommandPool");
        const VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                     nullptr, commandPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        check(vkAllocateCommandBuffers(device_, &allocation, &commandBuffer_), "vkAllocateCommandBuffers");
        const VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device_, &fence, nullptr, &fence_), "vkCreateFence");
        const VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                          VK_QUERY_TYPE_TIMESTAMP, 2, 0};
        check(vkCreateQueryPool(device_, &query, nullptr, &queryPool_), "vkCreateQueryPool");
    }

    void begin() {
        check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer");
        const VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        check(vkBeginCommandBuffer(commandBuffer_, &info), "vkBeginCommandBuffer");
    }

    void submit() {
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        const VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
                                1, &commandBuffer_, 0, nullptr};
        check(vkQueueSubmit(queue_, 1, &info, fence_), "vkQueueSubmit");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, std::numeric_limits<uint64_t>::max()),
              "vkWaitForFences");
        check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    }

    uint32_t width_, height_, mode_, dstWidth_, dstHeight_;
    uint32_t tileSrcW_, tileSrcH_, tileDstW_, tileDstH_;
    VkDeviceSize srcPlaneBytes_, dstPlaneBytes_;
    VkInstance instance_ = VK_NULL_HANDLE; VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{}; VkPhysicalDeviceMemoryProperties memoryProperties_{};
    uint32_t queueFamily_ = 0; VkDevice device_ = VK_NULL_HANDLE; VkQueue queue_ = VK_NULL_HANDLE;
    Buffer deviceOutput_{}; std::array<Buffer, 3> deviceInput_{};
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE; VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE; VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 3> descriptorSets_{}; VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE; VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
};

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_PreprocessingTest_nativeRunDownsample(
    JNIEnv* env, jclass, jobject bitmap, jbyteArray outputArray,
    jint width, jint height, jint mode, jbyteArray shaderArray, jint benchmarkRuns) {
    try {
        if ((mode != 1 && mode != 3 && mode != 4) || width <= 0 || height <= 0 || benchmarkRuns <= 0)
            throw std::runtime_error("Invalid arguments");
        AndroidBitmapInfo bitmapInfo{};
        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS ||
            bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
            static_cast<uint32_t>(width) > bitmapInfo.width ||
            static_cast<uint32_t>(height) > bitmapInfo.height)
            throw std::runtime_error("Bitmap must be RGBA_8888 and cover trimmed dimensions");
        void* bitmapPixels = nullptr;
        if (AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) != ANDROID_BITMAP_RESULT_SUCCESS)
            throw std::runtime_error("AndroidBitmap_lockPixels failed");
        VulkanDownsample downsample(static_cast<const uint8_t*>(bitmapPixels), bitmapInfo.stride,
                                    width, height, mode, bytesFromJava(env, shaderArray));
        AndroidBitmap_unlockPixels(env, bitmap);
        downsample.execute(1, false, false);
        const double secondRunMs = downsample.execute(1, true, false);
        const double totalMs = downsample.execute(static_cast<uint32_t>(benchmarkRuns), true, true);
        const auto output = downsample.output();
        if (env->GetArrayLength(outputArray) != static_cast<jsize>(output.size()))
            throw std::runtime_error("Output array has wrong size");
        env->SetByteArrayRegion(outputArray, 0, static_cast<jsize>(output.size()),
                                reinterpret_cast<const jbyte*>(output.data()));
        std::ostringstream report;
        report << downsample.deviceName() << '\n' << width << 'x' << height << " -> "
               << downsample.dstWidth() << 'x' << downsample.dstHeight() << '\n'
               << "Device-local YUV444, three dispatches per batch\n"
               << "Warm-up: 1 unmeasured batch\n" << std::fixed << std::setprecision(6)
               << "Second batch: " << secondRunMs << " ms\n"
               << benchmarkRuns << "-run average: " << totalMs / benchmarkRuns << " ms";
        return env->NewStringUTF(report.str().c_str());
    } catch (const std::exception& error) {
        return env->NewStringUTF((std::string("Error: ") + error.what()).c_str());
    }
}

