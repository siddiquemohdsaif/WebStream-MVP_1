#include <jni.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <sstream>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VulkanFeatureTest", __VA_ARGS__)

namespace {

const char* yesNo(bool value) {
    return value ? "YES" : "NO";
}

void appendUsage(std::ostringstream& out, VkImageUsageFlags flags, VkImageUsageFlagBits bit,
                 const char* name) {
    out << "  " << name << ": " << yesNo((flags & bit) != 0) << "\n";
}

std::string resultWithError(const std::string& message) {
    LOGI("%s", message.c_str());
    return "Error: " + message;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_VulkanFeatureTestActivity_nativeCheckVulkanFeatures(
        JNIEnv* env,
        jclass,
        jobject surface) {
    if (!surface) {
        const std::string result = resultWithError("Surface is null");
        return env->NewStringUTF(result.c_str());
    }

    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (!window) {
        const std::string result = resultWithError("ANativeWindow_fromSurface failed");
        return env->NewStringUTF(result.c_str());
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR vkSurface = VK_NULL_HANDLE;
    std::ostringstream out;

    const auto cleanup = [&]() {
        if (vkSurface) vkDestroySurfaceKHR(instance, vkSurface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        ANativeWindow_release(window);
    };

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "VulkanFeatureTest";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    const char* instanceExts[] = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = 2;
    instanceInfo.ppEnabledExtensionNames = instanceExts;

    VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        ANativeWindow_release(window);
        const std::string text = resultWithError("vkCreateInstance failed: " + std::to_string(result));
        return env->NewStringUTF(text.c_str());
    }

    VkAndroidSurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    surfaceInfo.window = window;
    result = vkCreateAndroidSurfaceKHR(instance, &surfaceInfo, nullptr, &vkSurface);
    if (result != VK_SUCCESS) {
        cleanup();
        const std::string text = resultWithError("vkCreateAndroidSurfaceKHR failed: " + std::to_string(result));
        return env->NewStringUTF(text.c_str());
    }

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS || deviceCount == 0) {
        cleanup();
        const std::string text = resultWithError("No Vulkan physical devices found");
        return env->NewStringUTF(text.c_str());
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    out << "Vulkan surface/swapchain feature check\n\n";
    out << "Physical devices: " << deviceCount << "\n\n";

    for (uint32_t deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        VkPhysicalDevice device = devices[deviceIndex];
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);

        out << "Device " << deviceIndex << ": " << props.deviceName << "\n";
        out << "API: "
            << VK_VERSION_MAJOR(props.apiVersion) << "."
            << VK_VERSION_MINOR(props.apiVersion) << "."
            << VK_VERSION_PATCH(props.apiVersion) << "\n";

        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());

        int selectedQueueFamily = -1;
        for (uint32_t q = 0; q < queueCount; ++q) {
            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, q, vkSurface, &presentSupported);
            const bool graphics = (queues[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            if (graphics && presentSupported) {
                selectedQueueFamily = static_cast<int>(q);
                break;
            }
        }

        if (selectedQueueFamily < 0) {
            out << "No graphics+present queue for this surface.\n\n";
            continue;
        }

        VkSurfaceCapabilitiesKHR caps{};
        result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, vkSurface, &caps);
        if (result != VK_SUCCESS) {
            out << "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: " << result << "\n\n";
            continue;
        }

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkSurface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        if (formatCount > 0) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkSurface, &formatCount, formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, vkSurface, &presentModeCount, nullptr);

        out << "Selected queue family: " << selectedQueueFamily << "\n";
        out << "Surface extent: " << caps.currentExtent.width << "x" << caps.currentExtent.height << "\n";
        out << "Min images: " << caps.minImageCount << "\n";
        out << "Max images: " << caps.maxImageCount << " (0 means no fixed max)\n";
        out << "Surface formats: " << formatCount << "\n";
        if (formatCount > 0) {
            out << "First format: " << formats[0].format
                << " colorSpace=" << formats[0].colorSpace << "\n";
        }
        out << "Present modes: " << presentModeCount << "\n\n";

        out << "Swapchain supportedUsageFlags = 0x"
            << std::hex << caps.supportedUsageFlags << std::dec << "\n";
        appendUsage(out, caps.supportedUsageFlags, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    "COLOR_ATTACHMENT_BIT");
        appendUsage(out, caps.supportedUsageFlags, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    "TRANSFER_DST_BIT");
        appendUsage(out, caps.supportedUsageFlags, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    "TRANSFER_SRC_BIT");
        appendUsage(out, caps.supportedUsageFlags, VK_IMAGE_USAGE_SAMPLED_BIT,
                    "SAMPLED_BIT");
        appendUsage(out, caps.supportedUsageFlags, VK_IMAGE_USAGE_STORAGE_BIT,
                    "STORAGE_BIT");

        out << "\nAnswer:\n";
        out << "  Compute shader direct imageStore() to swapchain: "
            << yesNo((caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0) << "\n";
        out << "  Compute to RGBA image then vkCmdCopyImage to swapchain: "
            << yesNo((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0) << "\n";
        out << "  Fragment shader render pass to swapchain: "
            << yesNo((caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0) << "\n";

        out << "\n";
    }

    const std::string text = out.str();
    LOGI("%s", text.c_str());
    cleanup();
    return env->NewStringUTF(text.c_str());
}
