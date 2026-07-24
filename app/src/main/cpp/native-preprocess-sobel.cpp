#include <jni.h>
#include <vulkan/vulkan.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SobelVulkan", __VA_ARGS__)

constexpr uint32_t kLocalSize = 32;
constexpr uint32_t kPixelsPerInvocation = 16;
constexpr uint32_t kPixelsPerWorkgroup = kLocalSize * kPixelsPerInvocation;
constexpr uint32_t kBenchmarkIterations = 1;

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: "
                                 + std::to_string(result));
    }
}

uint32_t roundUp(uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t inputStrideWords;
    uint32_t outputStrideWords;
    uint32_t inputPlaneWords;
    uint32_t outputPlaneWords;
};

class VulkanSobel {
public:
    VulkanSobel(const std::vector<uint8_t>& input, uint32_t width, uint32_t height,
                const std::vector<uint8_t>& shader)
        : width_(width), height_(height),
          pixelCount_(static_cast<size_t>(width) * height) {
        createInstance();
        selectDevice();
        createDevice();
        createBuffers(input);
        createPipeline(shader);
        createCommands();
        uploadInput();
    }

    ~VulkanSobel() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            if (queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
            if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            if (descriptorLayout_) vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
            destroyBuffer(stagingOutput_);
            destroyBuffer(deviceOutput_);
            destroyBuffer(deviceInput_);
            destroyBuffer(stagingInput_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    void warmUp() {
        resetAndBegin();
        bindAndDispatch();
        endSubmitWait();
    }

    double benchmark(uint32_t iterations, bool copyOutput) {
        resetAndBegin();
        vkCmdResetQueryPool(commandBuffer_, queryPool_, 0, 2);
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            queryPool_, 0);
        bindPipeline();

        const VkMemoryBarrier repeatBarrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT};
        for (uint32_t i = 0; i < iterations; ++i) {
            vkCmdDispatch(commandBuffer_, workgroupsX_, height_, 3);
            if (i + 1 < iterations) {
                vkCmdPipelineBarrier(commandBuffer_,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &repeatBarrier, 0, nullptr, 0, nullptr);
            }
        }
        vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            queryPool_, 1);

        if (copyOutput) {
            const VkBufferMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                deviceOutput_.buffer, 0, deviceOutput_.size};
            vkCmdPipelineBarrier(commandBuffer_,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 1, &barrier, 0, nullptr);
            const VkBufferCopy copy{0, 0, deviceOutput_.size};
            vkCmdCopyBuffer(commandBuffer_, deviceOutput_.buffer,
                            stagingOutput_.buffer, 1, &copy);
        }
        endSubmitWait();

        std::array<uint64_t, 2> timestamps{};
        check(vkGetQueryPoolResults(device_, queryPool_, 0, 2,
                                    sizeof(timestamps), timestamps.data(),
                                    sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults");
        return static_cast<double>(timestamps[1] - timestamps[0])
               * properties_.limits.timestampPeriod / 1'000'000.0;
    }

    std::vector<uint8_t> readOutput() {
        std::vector<uint32_t> values(static_cast<size_t>(outputPlaneWords_) * 3u);
        void* mapped = nullptr;
        check(vkMapMemory(device_, stagingOutput_.memory, 0,
                          stagingOutput_.size, 0, &mapped),
              "vkMapMemory(output)");
        std::memcpy(values.data(), mapped, static_cast<size_t>(stagingOutput_.size));
        vkUnmapMemory(device_, stagingOutput_.memory);
        std::vector<uint8_t> output(pixelCount_ * 3u);
        for (uint32_t channel = 0; channel < 3u; ++channel) {
            for (uint32_t y = 0; y < height_; ++y) {
                for (uint32_t x = 0; x < width_; ++x) {
                    const uint32_t word = values[channel * outputPlaneWords_
                        + static_cast<size_t>(y) * outputStrideWords_ + (x >> 2u)];
                    output[channel * pixelCount_ + static_cast<size_t>(y) * width_ + x] =
                        static_cast<uint8_t>((word >> ((x & 3u) * 8u)) & 255u);
                }
            }
        }
        return output;
    }

    const char* deviceName() const { return properties_.deviceName; }
    uint32_t apiVersion() const { return properties_.apiVersion; }

private:
    void createInstance() {
        const VkApplicationInfo appInfo{
            VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
            "Sobotest", 1, "Sobotest", 1, VK_API_VERSION_1_1};
        const VkInstanceCreateInfo info{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
            &appInfo, 0, nullptr, 0, nullptr};
        check(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        if (count == 0) throw std::runtime_error("No Vulkan GPU found");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
              "vkEnumeratePhysicalDevices");
        for (VkPhysicalDevice candidate : devices) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            for (uint32_t i = 0; i < familyCount; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                    families[i].timestampValidBits != 0) {
                    physicalDevice_ = candidate;
                    queueFamily_ = i;
                    vkGetPhysicalDeviceProperties(candidate, &properties_);
                    vkGetPhysicalDeviceMemoryProperties(candidate, &memoryProperties_);
                    return;
                }
            }
        }
        throw std::runtime_error("No timestamp-capable Vulkan compute queue found");
    }

    void createDevice() {
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queueInfo{
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
            queueFamily_, 1, &priority};
        const VkDeviceCreateInfo info{
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
            1, &queueInfo, 0, nullptr, 0, nullptr, nullptr};
        check(vkCreateDevice(physicalDevice_, &info, nullptr, &device_),
              "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    }

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) != 0 &&
                (memoryProperties_.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        throw std::runtime_error("No compatible Vulkan memory type");
    }

    Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags flags) {
        Buffer result{};
        result.size = size;
        const VkBufferCreateInfo info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
            size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        check(vkCreateBuffer(device_, &info, nullptr, &result.buffer),
              "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
            requirements.size,
            findMemoryType(requirements.memoryTypeBits, flags)};
        check(vkAllocateMemory(device_, &allocation, nullptr, &result.memory),
              "vkAllocateMemory");
        check(vkBindBufferMemory(device_, result.buffer, result.memory, 0),
              "vkBindBufferMemory");
        return result;
    }

    void destroyBuffer(Buffer& value) {
        if (value.buffer) vkDestroyBuffer(device_, value.buffer, nullptr);
        if (value.memory) vkFreeMemory(device_, value.memory, nullptr);
        value = {};
    }

    void createBuffers(const std::vector<uint8_t>& input) {
        const uint32_t outputSpanPixels = roundUp(width_, kPixelsPerWorkgroup);
        paddedWidth_ = outputSpanPixels + 4u;
        inputStrideWords_ = paddedWidth_ / 4u;
        outputStrideWords_ = outputSpanPixels / 4u;
        const uint32_t paddedHeight = height_ + 2u;
        inputPlaneWords_ = inputStrideWords_ * paddedHeight;
        outputPlaneWords_ = outputStrideWords_ * height_;
        std::vector<uint32_t> padded(
            static_cast<size_t>(inputPlaneWords_) * 3u, 0);
        auto setPacked = [&](uint32_t channel, uint32_t y, uint32_t x, uint8_t value) {
            uint32_t& word = padded[channel * inputPlaneWords_
                + static_cast<size_t>(y) * inputStrideWords_ + (x >> 2u)];
            const uint32_t shift = (x & 3u) * 8u;
            word = (word & ~(255u << shift)) | (static_cast<uint32_t>(value) << shift);
        };
        for (uint32_t channel = 0; channel < 3u; ++channel) {
            for (uint32_t y = 0; y < height_; ++y) {
                for (uint32_t x = 0; x < width_; ++x) {
                    setPacked(channel, y + 1u, x + 1u,
                              input[channel * pixelCount_ + static_cast<size_t>(y) * width_ + x]);
                }
            }
        }
        workgroupsX_ = (width_ + kPixelsPerWorkgroup - 1) / kPixelsPerWorkgroup;
        const VkDeviceSize inputBytes = padded.size() * sizeof(uint32_t);
        const VkDeviceSize outputBytes =
            static_cast<VkDeviceSize>(outputPlaneWords_) * 3u * sizeof(uint32_t);
        stagingInput_ = makeBuffer(
            inputBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        deviceInput_ = makeBuffer(
            inputBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        deviceOutput_ = makeBuffer(
            outputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        stagingOutput_ = makeBuffer(
            outputBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        check(vkMapMemory(device_, stagingInput_.memory, 0, inputBytes, 0, &mapped),
              "vkMapMemory(input)");
        std::memcpy(mapped, padded.data(), static_cast<size_t>(inputBytes));
        vkUnmapMemory(device_, stagingInput_.memory);
    }

    void createPipeline(const std::vector<uint8_t>& shaderBytes) {
        if (shaderBytes.empty() || shaderBytes.size() % 4 != 0) {
            throw std::runtime_error("Invalid SPIR-V asset");
        }
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        const VkDescriptorSetLayoutCreateInfo descriptorInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
            static_cast<uint32_t>(bindings.size()), bindings.data()};
        check(vkCreateDescriptorSetLayout(device_, &descriptorInfo, nullptr,
                                          &descriptorLayout_),
              "vkCreateDescriptorSetLayout");
        const VkPushConstantRange range{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        const VkPipelineLayoutCreateInfo layoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
            1, &descriptorLayout_, 1, &range};
        check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout");

        const VkShaderModuleCreateInfo moduleInfo{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            shaderBytes.size(),
            reinterpret_cast<const uint32_t*>(shaderBytes.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &moduleInfo, nullptr, &module),
              "vkCreateShaderModule");
        const VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        const VkComputePipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
            stage, pipelineLayout_, VK_NULL_HANDLE, -1};
        VkResult pipelineResult = vkCreateComputePipelines(
            device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, module, nullptr);
        check(pipelineResult, "vkCreateComputePipelines");

        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        const VkDescriptorPoolCreateInfo poolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0,
            1, 1, &poolSize};
        check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_),
              "vkCreateDescriptorPool");
        const VkDescriptorSetAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
            descriptorPool_, 1, &descriptorLayout_};
        check(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet_),
              "vkAllocateDescriptorSets");
        const std::array<VkDescriptorBufferInfo, 2> infos{{
            {deviceInput_.buffer, 0, deviceInput_.size},
            {deviceOutput_.buffer, 0, deviceOutput_.size}}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (uint32_t i = 0; i < writes.size(); ++i) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         descriptorSet_, i, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         nullptr, &infos[i], nullptr};
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    void createCommands() {
        const VkCommandPoolCreateInfo poolInfo{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamily_};
        check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
              "vkCreateCommandPool");
        const VkCommandBufferAllocateInfo commandInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
            commandPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        check(vkAllocateCommandBuffers(device_, &commandInfo, &commandBuffer_),
              "vkAllocateCommandBuffers");
        const VkFenceCreateInfo fenceInfo{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence");
        const VkQueryPoolCreateInfo queryInfo{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, 2, 0};
        check(vkCreateQueryPool(device_, &queryInfo, nullptr, &queryPool_),
              "vkCreateQueryPool");
    }

    void uploadInput() {
        resetAndBegin();
        const VkBufferCopy copy{0, 0, stagingInput_.size};
        vkCmdCopyBuffer(commandBuffer_, stagingInput_.buffer,
                        deviceInput_.buffer, 1, &copy);
        const VkBufferMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            deviceInput_.buffer, 0, deviceInput_.size};
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
        endSubmitWait();
    }

    void resetAndBegin() {
        check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer");
        const VkCommandBufferBeginInfo info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        check(vkBeginCommandBuffer(commandBuffer_, &info), "vkBeginCommandBuffer");
    }

    void bindPipeline() {
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
        const PushConstants constants{width_, height_, inputStrideWords_, outputStrideWords_,
                                      inputPlaneWords_, outputPlaneWords_};
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
    }

    void bindAndDispatch() {
        bindPipeline();
        vkCmdDispatch(commandBuffer_, workgroupsX_, height_, 3);
    }

    void endSubmitWait() {
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        const VkSubmitInfo submit{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
            0, nullptr, nullptr, 1, &commandBuffer_, 0, nullptr};
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE,
                              std::numeric_limits<uint64_t>::max()),
              "vkWaitForFences");
        check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    }

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    size_t pixelCount_ = 0;
    uint32_t paddedWidth_ = 0;
    uint32_t inputStrideWords_ = 0;
    uint32_t outputStrideWords_ = 0;
    uint32_t inputPlaneWords_ = 0;
    uint32_t outputPlaneWords_ = 0;
    uint32_t workgroupsX_ = 0;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    uint32_t queueFamily_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Buffer stagingInput_{};
    Buffer deviceInput_{};
    Buffer deviceOutput_{};
    Buffer stagingOutput_{};
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
};

std::vector<uint8_t> getBytes(JNIEnv* env, jbyteArray array) {
    const jsize size = env->GetArrayLength(array);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    env->GetByteArrayRegion(array, 0, size,
                            reinterpret_cast<jbyte*>(bytes.data()));
    return bytes;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_PreprocessingTest_nativeRunSobel(
    JNIEnv* env, jobject, jbyteArray inputArray, jbyteArray outputArray,
    jint width, jint height, jbyteArray shaderArray) {
    try {
        const auto input = getBytes(env, inputArray);
        const auto shader = getBytes(env, shaderArray);
        if (width <= 0 || height <= 0 ||
            input.size() != static_cast<size_t>(width) * height * 3u) {
            throw std::runtime_error("Invalid input dimensions");
        }
        if (env->GetArrayLength(outputArray) != static_cast<jsize>(input.size())) {
            throw std::runtime_error("Invalid output array size");
        }

        VulkanSobel sobel(input, static_cast<uint32_t>(width),
                          static_cast<uint32_t>(height), shader);
        LOGI("Vulkan setup complete");
        sobel.warmUp();
        LOGI("Warm-up complete");
        const double oneMs = sobel.benchmark(1, false);
        LOGI("One-dispatch benchmark complete: %.6f ms", oneMs);
        LOGI("Starting %u-dispatch benchmark", kBenchmarkIterations);
        double repeatedMs = 0.0;
        for (uint32_t i = 0; i < kBenchmarkIterations; ++i) {
            repeatedMs += sobel.benchmark(1, i + 1 == kBenchmarkIterations);
        }
        LOGI("%u-dispatch benchmark complete: %.6f ms",
             kBenchmarkIterations, repeatedMs);
        const auto output = sobel.readOutput();
        env->SetByteArrayRegion(outputArray, 0, static_cast<jsize>(output.size()),
                                reinterpret_cast<const jbyte*>(output.data()));

        uint64_t checksum = 1469598103934665603ull;
        for (uint8_t value : output) {
            checksum = (checksum ^ value) * 1099511628211ull;
        }
        const double averageMs = repeatedMs / kBenchmarkIterations;
        const double gpixels = static_cast<double>(input.size()) / averageMs / 1'000'000.0;
        const uint32_t api = sobel.apiVersion();
        std::ostringstream report;
        report << sobel.deviceName() << "\n"
               << "Vulkan " << VK_VERSION_MAJOR(api) << '.' << VK_VERSION_MINOR(api)
               << '.' << VK_VERSION_PATCH(api) << "\n"
               << width << 'x' << height << " x 3 Y/U/V planes = "
               << input.size() << " channel-pixels\n"
               << "32 invocations/workgroup, 16 pixels/invocation\n"
               << "one contiguous packed uint32 input/output: [Y][U][V]\n\n"
               << std::fixed << std::setprecision(6)
               << "1 dispatch: " << oneMs << " ms\n"
               << kBenchmarkIterations << " dispatches: " << repeatedMs << " ms\n"
               << "Average: " << averageMs << " ms/dispatch\n"
               << std::setprecision(3)
               << "Throughput: " << gpixels << " GPixels/s\n"
               << "Output FNV-1a: 0x" << std::hex << checksum;
        return env->NewStringUTF(report.str().c_str());
    } catch (const std::exception& error) {
        const std::string message = std::string("Error: ") + error.what();
        return env->NewStringUTF(message.c_str());
    }
}




