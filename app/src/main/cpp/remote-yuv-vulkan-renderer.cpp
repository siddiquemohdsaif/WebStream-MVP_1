#include "remote-yuv-vulkan-renderer.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#define LOGR(...) __android_log_print(ANDROID_LOG_INFO, "RemoteYuvVulkan", __VA_ARGS__)
#define LOGRE(...) __android_log_print(ANDROID_LOG_ERROR, "RemoteYuvVulkan", __VA_ARGS__)

namespace {

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkDeviceSize size{};
};

struct Texture {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    int width{};
    int height{};
};

struct RemoteRenderer {
    VkInstance instance{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    uint32_t queueFamily{};
    VkQueue queue{};
    VkSwapchainKHR swapchain{};
    VkFormat swapFormat{};
    VkExtent2D extent{};
    VkSurfaceTransformFlagBitsKHR surfaceTransform{VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR};
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    std::vector<VkFramebuffer> framebuffers;
    VkRenderPass renderPass{};
    VkDescriptorSetLayout setLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptorPool{};
    VkDescriptorSet descriptor{};
    VkSampler sampler{};
    VkCommandPool commandPool{};
    VkCommandBuffer command{};
    VkSemaphore acquired{};
    VkSemaphore finished{};
    VkFence fence{};
    std::array<Texture, 3> textures{};
    Buffer staging{};
    VkDeviceSize stagingSize{};
    AAssetManager* assets{};
    bool ready{};
};

std::mutex gMutex;
RemoteRenderer r;

bool ok(VkResult result, const char* label) {
    if (result == VK_SUCCESS) return true;
    LOGRE("%s failed: %d", label, result);
    return false;
}

double ms(std::chrono::steady_clock::time_point start,
          std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<uint8_t> readAsset(const char* name) {
    std::vector<uint8_t> out;
    if (!r.assets) return out;
    AAsset* asset = AAssetManager_open(r.assets, name, AASSET_MODE_BUFFER);
    if (!asset) {
        LOGRE("asset open failed: %s", name);
        return out;
    }
    const off_t length = AAsset_getLength(asset);
    out.resize(static_cast<size_t>(length));
    AAsset_read(asset, out.data(), static_cast<size_t>(length));
    AAsset_close(asset);
    return out;
}

uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(r.physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

void destroyBuffer(Buffer& b) {
    if (b.buffer) vkDestroyBuffer(r.device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(r.device, b.memory, nullptr);
    b = {};
}

bool createBuffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags flags, const char* label) {
    b.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!ok(vkCreateBuffer(r.device, &info, nullptr, &b.buffer), label)) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.device, b.buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memoryType(req.memoryTypeBits, flags);
    if (alloc.memoryTypeIndex == UINT32_MAX ||
        !ok(vkAllocateMemory(r.device, &alloc, nullptr, &b.memory), label)) {
        return false;
    }
    return ok(vkBindBufferMemory(r.device, b.buffer, b.memory, 0), label);
}

void destroyTexture(Texture& t) {
    if (t.view) vkDestroyImageView(r.device, t.view, nullptr);
    if (t.image) vkDestroyImage(r.device, t.image, nullptr);
    if (t.memory) vkFreeMemory(r.device, t.memory, nullptr);
    t = {};
}

void destroyFrameData() {
    for (auto& texture : r.textures) destroyTexture(texture);
    destroyBuffer(r.staging);
    r.stagingSize = 0;
}

void destroyAll() {
    if (!r.device && !r.instance) {
        r = {};
        return;
    }
    if (r.device) vkDeviceWaitIdle(r.device);
    destroyFrameData();
    if (r.fence) vkDestroyFence(r.device, r.fence, nullptr);
    if (r.acquired) vkDestroySemaphore(r.device, r.acquired, nullptr);
    if (r.finished) vkDestroySemaphore(r.device, r.finished, nullptr);
    if (r.commandPool) vkDestroyCommandPool(r.device, r.commandPool, nullptr);
    if (r.pipeline) vkDestroyPipeline(r.device, r.pipeline, nullptr);
    if (r.pipelineLayout) vkDestroyPipelineLayout(r.device, r.pipelineLayout, nullptr);
    if (r.descriptorPool) vkDestroyDescriptorPool(r.device, r.descriptorPool, nullptr);
    if (r.setLayout) vkDestroyDescriptorSetLayout(r.device, r.setLayout, nullptr);
    if (r.sampler) vkDestroySampler(r.device, r.sampler, nullptr);
    for (auto framebuffer : r.framebuffers) {
        if (framebuffer) vkDestroyFramebuffer(r.device, framebuffer, nullptr);
    }
    for (auto view : r.swapViews) {
        if (view) vkDestroyImageView(r.device, view, nullptr);
    }
    if (r.renderPass) vkDestroyRenderPass(r.device, r.renderPass, nullptr);
    if (r.swapchain) vkDestroySwapchainKHR(r.device, r.swapchain, nullptr);
    if (r.device) vkDestroyDevice(r.device, nullptr);
    if (r.surface) vkDestroySurfaceKHR(r.instance, r.surface, nullptr);
    if (r.instance) vkDestroyInstance(r.instance, nullptr);
    r = {};
}

VkShaderModule shaderModule(const char* name) {
    auto bytes = readAsset(name);
    if (bytes.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule module{};
    return ok(vkCreateShaderModule(r.device, &info, nullptr, &module), name)
           ? module
           : VK_NULL_HANDLE;
}

bool createTexture(Texture& t, int width, int height) {
    t.width = width;
    t.height = height;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_R8_UNORM;
    image.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!ok(vkCreateImage(r.device, &image, nullptr, &t.image), "remote texture image")) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(r.device, t.image, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (alloc.memoryTypeIndex == UINT32_MAX ||
        !ok(vkAllocateMemory(r.device, &alloc, nullptr, &t.memory), "remote texture memory")) {
        return false;
    }
    if (!ok(vkBindImageMemory(r.device, t.image, t.memory, 0), "remote texture bind")) return false;
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = t.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R8_UNORM;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return ok(vkCreateImageView(r.device, &view, nullptr, &t.view), "remote texture view");
}

bool createYuvFrameResources(int width, int height) {
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    if (r.textures[0].width == width && r.textures[0].height == height &&
        r.textures[1].width == chromaWidth && r.textures[1].height == chromaHeight) {
        return true;
    }
    destroyFrameData();
    if (!createTexture(r.textures[0], width, height) ||
        !createTexture(r.textures[1], chromaWidth, chromaHeight) ||
        !createTexture(r.textures[2], chromaWidth, chromaHeight)) {
        return false;
    }
    r.stagingSize = static_cast<VkDeviceSize>(width) * height +
                    static_cast<VkDeviceSize>(chromaWidth) * chromaHeight * 2;
    if (!createBuffer(r.staging, r.stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      "remote yuv staging")) {
        return false;
    }
    std::array<VkDescriptorImageInfo, 3> imageInfo{};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < 3; ++i) {
        imageInfo[i] = {r.sampler, r.textures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = r.descriptor;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imageInfo[i];
    }
    vkUpdateDescriptorSets(r.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

bool initRenderer(ANativeWindow* window, AAssetManager* assets) {
    r.assets = assets;
    const char* instanceExtensions[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance.pApplicationInfo = &app;
    instance.enabledExtensionCount = 2;
    instance.ppEnabledExtensionNames = instanceExtensions;
    if (!ok(vkCreateInstance(&instance, nullptr, &r.instance), "remote instance")) return false;

    VkAndroidSurfaceCreateInfoKHR surface{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    surface.window = window;
    if (!ok(vkCreateAndroidSurfaceKHR(r.instance, &surface, nullptr, &r.surface), "remote surface")) {
        return false;
    }

    uint32_t physicalCount = 0;
    vkEnumeratePhysicalDevices(r.instance, &physicalCount, nullptr);
    std::vector<VkPhysicalDevice> physicals(physicalCount);
    vkEnumeratePhysicalDevices(r.instance, &physicalCount, physicals.data());
    for (auto physical : physicals) {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
        for (uint32_t i = 0; i < familyCount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, r.surface, &present);
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                r.physical = physical;
                r.queueFamily = i;
                break;
            }
        }
        if (r.physical) break;
    }
    if (!r.physical) return false;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = r.queueFamily;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const char* deviceExtensions[] = {"VK_KHR_swapchain"};
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = 1;
    device.ppEnabledExtensionNames = deviceExtensions;
    if (!ok(vkCreateDevice(r.physical, &device, nullptr, &r.device), "remote device")) return false;
    vkGetDeviceQueue(r.device, r.queueFamily, 0, &r.queue);

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(r.physical, r.surface, &caps);
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(r.physical, r.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(r.physical, r.surface, &formatCount, formats.data());
    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (auto candidate : formats) {
        if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM ||
            candidate.format == VK_FORMAT_B8G8R8A8_UNORM) {
            surfaceFormat = candidate;
            break;
        }
    }
    r.swapFormat = surfaceFormat.format;
    r.extent = caps.currentExtent;
    if (r.extent.width == UINT32_MAX) {
        r.extent = {static_cast<uint32_t>(ANativeWindow_getWidth(window)),
                    static_cast<uint32_t>(ANativeWindow_getHeight(window))};
    }
    r.surfaceTransform = caps.currentTransform;
    uint32_t imageCount = std::max(caps.minImageCount + 1, 2u);
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);
    VkSwapchainCreateInfoKHR swap{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swap.surface = r.surface;
    swap.minImageCount = imageCount;
    swap.imageFormat = r.swapFormat;
    swap.imageColorSpace = surfaceFormat.colorSpace;
    swap.imageExtent = r.extent;
    swap.imageArrayLayers = 1;
    swap.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swap.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap.preTransform = caps.currentTransform;
    swap.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swap.clipped = VK_TRUE;
    if (!ok(vkCreateSwapchainKHR(r.device, &swap, nullptr, &r.swapchain), "remote swapchain")) {
        return false;
    }
    vkGetSwapchainImagesKHR(r.device, r.swapchain, &imageCount, nullptr);
    r.swapImages.resize(imageCount);
    vkGetSwapchainImagesKHR(r.device, r.swapchain, &imageCount, r.swapImages.data());
    r.swapViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = r.swapImages[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = r.swapFormat;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!ok(vkCreateImageView(r.device, &view, nullptr, &r.swapViews[i]), "remote swap view")) {
            return false;
        }
    }

    VkAttachmentDescription color{};
    color.format = r.swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPass.attachmentCount = 1;
    renderPass.pAttachments = &color;
    renderPass.subpassCount = 1;
    renderPass.pSubpasses = &subpass;
    if (!ok(vkCreateRenderPass(r.device, &renderPass, nullptr, &r.renderPass), "remote render pass")) {
        return false;
    }
    r.framebuffers.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageView attachments[] = {r.swapViews[i]};
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = r.renderPass;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = attachments;
        framebuffer.width = r.extent.width;
        framebuffer.height = r.extent.height;
        framebuffer.layers = 1;
        if (!ok(vkCreateFramebuffer(r.device, &framebuffer, nullptr, &r.framebuffers[i]), "remote framebuffer")) {
            return false;
        }
    }

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    setInfo.pBindings = bindings.data();
    if (!ok(vkCreateDescriptorSetLayout(r.device, &setInfo, nullptr, &r.setLayout), "remote set layout")) {
        return false;
    }

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(float) * 2 + sizeof(int32_t) * 2;
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &r.setLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (!ok(vkCreatePipelineLayout(r.device, &layout, nullptr, &r.pipelineLayout), "remote pipeline layout")) {
        return false;
    }

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &poolSize;
    if (!ok(vkCreateDescriptorPool(r.device, &pool, nullptr, &r.descriptorPool), "remote descriptor pool")) {
        return false;
    }
    VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAlloc.descriptorPool = r.descriptorPool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &r.setLayout;
    if (!ok(vkAllocateDescriptorSets(r.device, &setAlloc, &r.descriptor), "remote descriptor set")) {
        return false;
    }

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = 1.0f;
    if (!ok(vkCreateSampler(r.device, &sampler, nullptr, &r.sampler), "remote sampler")) return false;

    VkShaderModule vert = shaderModule("camera.vert.spv");
    VkShaderModule frag = shaderModule("camera.frag.spv");
    if (!vert || !frag) return false;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = r.pipelineLayout;
    pipeline.renderPass = r.renderPass;
    bool pipelineOk = ok(vkCreateGraphicsPipelines(r.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &r.pipeline),
                         "remote graphics pipeline");
    vkDestroyShaderModule(r.device, vert, nullptr);
    vkDestroyShaderModule(r.device, frag, nullptr);
    if (!pipelineOk) return false;

    VkCommandPoolCreateInfo commandPool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPool.queueFamilyIndex = r.queueFamily;
    if (!ok(vkCreateCommandPool(r.device, &commandPool, nullptr, &r.commandPool), "remote command pool")) {
        return false;
    }
    VkCommandBufferAllocateInfo commandAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandAlloc.commandPool = r.commandPool;
    commandAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandAlloc.commandBufferCount = 1;
    if (!ok(vkAllocateCommandBuffers(r.device, &commandAlloc, &r.command), "remote command buffer")) return false;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(r.device, &semaphore, nullptr, &r.acquired);
    vkCreateSemaphore(r.device, &semaphore, nullptr, &r.finished);
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(r.device, &fence, nullptr, &r.fence);
    r.ready = true;
    LOGR("remote Vulkan ready swapchain=%ux%u", r.extent.width, r.extent.height);
    return true;
}

void copyPlaneToTexture(uint32_t plane, VkDeviceSize offset, int width, int height) {
    Texture& texture = r.textures[plane];
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(r.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.bufferOffset = offset;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    vkCmdCopyBufferToImage(r.command, r.staging.buffer, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(r.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}  // namespace

std::string remoteVulkanSetWindow(ANativeWindow* window, AAssetManager* assets) {
    std::lock_guard<std::mutex> lock(gMutex);
    destroyAll();
    if (!window || !assets) return "Remote Vulkan surface released";
    if (!initRenderer(window, assets)) {
        destroyAll();
        return "Error: remote Vulkan initialization failed";
    }
    std::ostringstream out;
    out << "Remote Vulkan SurfaceView ready | swapchain " << r.extent.width << "x" << r.extent.height;
    return out.str();
}

void remoteVulkanDestroy() {
    std::lock_guard<std::mutex> lock(gMutex);
    destroyAll();
}

bool remoteVulkanRenderYuv420(const uint8_t* yuv420,
                              size_t yuv420Size,
                              int width,
                              int height,
                              uint16_t rotation,
                              bool mirror) {
    const auto totalStart = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(gMutex);
    if (!r.ready || width <= 0 || height <= 0) return false;
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    const size_t ySize = static_cast<size_t>(width) * height;
    const size_t uvSize = static_cast<size_t>(chromaWidth) * chromaHeight;
    const size_t requiredYuvSize = ySize + uvSize + uvSize;
    if (!yuv420 || yuv420Size < requiredYuvSize) return false;
    const auto resourcesStart = std::chrono::steady_clock::now();
    if (!createYuvFrameResources(width, height)) return false;
    const auto resourcesEnd = std::chrono::steady_clock::now();
    const auto previousFenceStart = std::chrono::steady_clock::now();
    if (vkWaitForFences(r.device, 1, &r.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;
    const auto previousFenceEnd = std::chrono::steady_clock::now();

    const auto stagingStart = std::chrono::steady_clock::now();
    void* mapped{};
    if (vkMapMemory(r.device, r.staging.memory, 0, r.staging.size, 0, &mapped) != VK_SUCCESS) return false;
    auto* bytes = static_cast<uint8_t*>(mapped);
    std::memcpy(bytes, yuv420, requiredYuvSize);
    vkUnmapMemory(r.device, r.staging.memory);
    const auto stagingEnd = std::chrono::steady_clock::now();

    uint32_t imageIndex = 0;
    const auto acquireStart = std::chrono::steady_clock::now();
    VkResult acquire = vkAcquireNextImageKHR(r.device, r.swapchain, UINT64_MAX,
                                             r.acquired, VK_NULL_HANDLE, &imageIndex);
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) return false;
    const auto acquireEnd = std::chrono::steady_clock::now();
    const auto recordStart = std::chrono::steady_clock::now();
    vkResetFences(r.device, 1, &r.fence);
    vkResetCommandBuffer(r.command, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(r.command, &begin) != VK_SUCCESS) return false;

    const auto uploadRecordStart = std::chrono::steady_clock::now();
    copyPlaneToTexture(0, 0, width, height);
    copyPlaneToTexture(1, static_cast<VkDeviceSize>(ySize), chromaWidth, chromaHeight);
    copyPlaneToTexture(2, static_cast<VkDeviceSize>(ySize + uvSize), chromaWidth, chromaHeight);
    const auto uploadRecordEnd = std::chrono::steady_clock::now();

    const auto drawRecordStart = std::chrono::steady_clock::now();
    VkClearValue clear{{{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPass.renderPass = r.renderPass;
    renderPass.framebuffer = r.framebuffers[imageIndex];
    renderPass.renderArea = {{0, 0}, r.extent};
    renderPass.clearValueCount = 1;
    renderPass.pClearValues = &clear;
    vkCmdBeginRenderPass(r.command, &renderPass, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(r.extent.width), static_cast<float>(r.extent.height),
                        0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, r.extent};
    vkCmdSetViewport(r.command, 0, 1, &viewport);
    vkCmdSetScissor(r.command, 0, 1, &scissor);
    vkCmdBindPipeline(r.command, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipeline);
    vkCmdBindDescriptorSets(r.command, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipelineLayout,
                            0, 1, &r.descriptor, 0, nullptr);
    struct Push {
        float scale[2];
        int32_t rotation;
        int32_t mirror;
    } push{{1.0f, 1.0f}, static_cast<int32_t>(rotation), mirror ? 1 : 0};
    const bool imageQuarterTurn = rotation == 90 || rotation == 270;
    const float imageAspect = imageQuarterTurn
            ? static_cast<float>(height) / static_cast<float>(width)
            : static_cast<float>(width) / static_cast<float>(height);
    const bool surfaceQuarterTurn =
            r.surfaceTransform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
            r.surfaceTransform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ||
            r.surfaceTransform == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR ||
            r.surfaceTransform == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR;
    const float screenAspect = surfaceQuarterTurn
            ? static_cast<float>(r.extent.height) / static_cast<float>(r.extent.width)
            : static_cast<float>(r.extent.width) / static_cast<float>(r.extent.height);
    if (imageAspect > screenAspect) push.scale[1] = screenAspect / imageAspect;
    else push.scale[0] = imageAspect / screenAspect;
    vkCmdPushConstants(r.command, r.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDraw(r.command, 3, 1, 0, 0);
    vkCmdEndRenderPass(r.command);
    const auto drawRecordEnd = std::chrono::steady_clock::now();
    vkEndCommandBuffer(r.command);
    const auto recordEnd = std::chrono::steady_clock::now();

    const auto submitStart = std::chrono::steady_clock::now();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &r.acquired;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &r.command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &r.finished;
    if (vkQueueSubmit(r.queue, 1, &submit, r.fence) != VK_SUCCESS) return false;
    const auto submitEnd = std::chrono::steady_clock::now();
    const auto presentStart = std::chrono::steady_clock::now();
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &r.finished;
    present.swapchainCount = 1;
    present.pSwapchains = &r.swapchain;
    present.pImageIndices = &imageIndex;
    vkQueuePresentKHR(r.queue, &present);
    const auto presentEnd = std::chrono::steady_clock::now();
    const auto fenceWaitStart = std::chrono::steady_clock::now();
    vkWaitForFences(r.device, 1, &r.fence, VK_TRUE, UINT64_MAX);
    const auto fenceWaitEnd = std::chrono::steady_clock::now();
    const auto totalEnd = std::chrono::steady_clock::now();
    LOGR("Remote Vulkan Frame: resourcesMs=%.3f waitPreviousMs=%.3f stagingMapCopyMs=%.3f acquireMs=%.3f recordTotalMs=%.3f recordUploadCmdMs=%.3f recordDrawCmdMs=%.3f submitMs=%.3f presentMs=%.3f gpuFenceWaitMs=%.3f totalMs=%.3f yBytes=%zu uvBytesEach=%zu size=%dx%d rotation=%u mirror=%s",
         ms(resourcesStart, resourcesEnd),
         ms(previousFenceStart, previousFenceEnd),
         ms(stagingStart, stagingEnd),
         ms(acquireStart, acquireEnd),
         ms(recordStart, recordEnd),
         ms(uploadRecordStart, uploadRecordEnd),
         ms(drawRecordStart, drawRecordEnd),
         ms(submitStart, submitEnd),
         ms(presentStart, presentEnd),
         ms(fenceWaitStart, fenceWaitEnd),
         ms(totalStart, totalEnd),
         ySize,
         uvSize,
         width,
         height,
         rotation,
         mirror ? "true" : "false");
    return true;
}
