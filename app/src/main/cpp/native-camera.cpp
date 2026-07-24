#include <jni.h>
#include <android/log.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include "native-vulkan-renderer.h"
#include "camera_transformation_evaluator.h"
#include "camera_preprocess_pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NativeCamera", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NativeCamera", __VA_ARGS__)

namespace {

enum class CameraPipelineFormat {
    Yuv420,
    Yuv444,
};

// Change only this line to switch the live camera preprocessing path.
constexpr CameraPipelineFormat kCameraPipelineFormat = CameraPipelineFormat::Yuv420;

// Change only these two lines to switch the live camera output resolution.
// Supported targets: 2560x1440, 1920x1080, 1280x720, 854x480, 640x360.
constexpr int kCameraDesiredOutputWidth = 1280;
constexpr int kCameraDesiredOutputHeight = 720;

class NativeYuvCamera {
public:
    ~NativeYuvCamera() { stop(); }

    std::string start(bool front, int requestedWidth, int requestedHeight) {
        stop();
        width_ = requestedWidth;
        height_ = requestedHeight;
        useFront_ = front;
        running_ = true;
        totalFrames_ = 0;
        fpsWindowFrames_ = 0;
        fpsWindowStartNs_ = 0;

        manager_ = ACameraManager_create();
        if (!manager_) return fail("ACameraManager_create failed");

        std::string cameraId;
        std::string cameraList = findCameraId(front, cameraId);
        if (cameraId.empty()) {
            return fail("No " + std::string(front ? "front" : "back") + " camera found. " + cameraList);
        }
        cameraId_ = cameraId;
        chooseSupportedYuvSize(cameraId_, requestedWidth, requestedHeight);
        if (!((width_ == 2560 && height_ == 1440) || (width_ == 1920 && height_ == 1080))) {
            return fail("Unsupported max 16:9 camera size " + std::to_string(width_) + "x" +
                        std::to_string(height_) + "; expected 2560x1440 or 1920x1080");
        }

        media_status_t readerStatus = AImageReader_new(
                width_, height_, AIMAGE_FORMAT_YUV_420_888, 4, &reader_);
        if (readerStatus != AMEDIA_OK || !reader_) {
            return fail("AImageReader_new YUV_420_888 failed: " + std::to_string(readerStatus));
        }
        AImageReader_ImageListener listener{};
        listener.context = this;
        listener.onImageAvailable = &NativeYuvCamera::onImageAvailable;
        AImageReader_setImageListener(reader_, &listener);

        media_status_t windowStatus = AImageReader_getWindow(reader_, &readerWindow_);
        if (windowStatus != AMEDIA_OK || !readerWindow_) {
            return fail("AImageReader_getWindow failed: " + std::to_string(windowStatus));
        }

        cameraCallbacks_.context = this;
        cameraCallbacks_.onDisconnected = &NativeYuvCamera::onDisconnected;
        cameraCallbacks_.onError = &NativeYuvCamera::onError;
        camera_status_t openStatus = ACameraManager_openCamera(
                manager_, cameraId_.c_str(), &cameraCallbacks_, &device_);
        if (openStatus != ACAMERA_OK || !device_) {
            return fail("ACameraManager_openCamera failed: " + std::to_string(openStatus));
        }

        camera_status_t outputContainerStatus = ACaptureSessionOutputContainer_create(&outputs_);
        if (outputContainerStatus != ACAMERA_OK || !outputs_) {
            return fail("ACaptureSessionOutputContainer_create failed: " + std::to_string(outputContainerStatus));
        }
        camera_status_t outputStatus = ACaptureSessionOutput_create(readerWindow_, &readerOutput_);
        if (outputStatus != ACAMERA_OK || !readerOutput_) {
            return fail("ACaptureSessionOutput_create failed: " + std::to_string(outputStatus));
        }
        ACaptureSessionOutputContainer_add(outputs_, readerOutput_);

        camera_status_t requestStatus = ACameraDevice_createCaptureRequest(
                device_, TEMPLATE_RECORD, &request_);
        if (requestStatus != ACAMERA_OK || !request_) {
            requestStatus = ACameraDevice_createCaptureRequest(device_, TEMPLATE_PREVIEW, &request_);
        }
        if (requestStatus != ACAMERA_OK || !request_) {
            return fail("ACameraDevice_createCaptureRequest failed: " + std::to_string(requestStatus));
        }
        configureBestFpsRange(cameraId_);

        camera_status_t targetStatus = ACameraOutputTarget_create(readerWindow_, &target_);
        if (targetStatus != ACAMERA_OK || !target_) {
            return fail("ACameraOutputTarget_create failed: " + std::to_string(targetStatus));
        }
        ACaptureRequest_addTarget(request_, target_);

        sessionCallbacks_.context = this;
        sessionCallbacks_.onClosed = &NativeYuvCamera::onSessionClosed;
        sessionCallbacks_.onReady = &NativeYuvCamera::onSessionReady;
        sessionCallbacks_.onActive = &NativeYuvCamera::onSessionActive;
        camera_status_t sessionStatus = ACameraDevice_createCaptureSession(
                device_, outputs_, &sessionCallbacks_, &session_);
        if (sessionStatus != ACAMERA_OK || !session_) {
            return fail("ACameraDevice_createCaptureSession failed: " + std::to_string(sessionStatus));
        }

        camera_status_t repeatStatus = ACameraCaptureSession_setRepeatingRequest(
                session_, nullptr, 1, &request_, nullptr);
        if (repeatStatus != ACAMERA_OK) {
            return fail("ACameraCaptureSession_setRepeatingRequest failed: " + std::to_string(repeatStatus));
        }

        camera_transform_set_camera(sensorOrientation_, front);
        camera_transform_start_logging();

        std::ostringstream out;
        out << "Native NDK camera ON | " << (front ? "front" : "back")
            << " id=" << cameraId_ << " | YUV_420_888 " << width_ << "x" << height_
            << " requested=" << requestedWidth << "x" << requestedHeight;
        LOGI("%s", out.str().c_str());
        return out.str();
    }

    void stop() {
        camera_transform_stop_logging();
        running_ = false;
        if (session_) {
            ACameraCaptureSession_stopRepeating(session_);
            ACameraCaptureSession_close(session_);
            session_ = nullptr;
        }
        if (request_ && target_) {
            ACaptureRequest_removeTarget(request_, target_);
        }
        if (target_) {
            ACameraOutputTarget_free(target_);
            target_ = nullptr;
        }
        if (request_) {
            ACaptureRequest_free(request_);
            request_ = nullptr;
        }
        if (readerOutput_) {
            ACaptureSessionOutput_free(readerOutput_);
            readerOutput_ = nullptr;
        }
        if (outputs_) {
            ACaptureSessionOutputContainer_free(outputs_);
            outputs_ = nullptr;
        }
        if (device_) {
            ACameraDevice_close(device_);
            device_ = nullptr;
        }
        if (reader_) {
            AImageReader_delete(reader_);
            reader_ = nullptr;
            readerWindow_ = nullptr;
        }
        if (manager_) {
            ACameraManager_delete(manager_);
            manager_ = nullptr;
        }
        cameraId_.clear();
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    std::string findCameraId(bool front, std::string& outId) {
        ACameraIdList* list = nullptr;
        camera_status_t status = ACameraManager_getCameraIdList(manager_, &list);
        if (status != ACAMERA_OK || !list) return "camera list unavailable";

        std::ostringstream seen;
        seen << "Seen cameras:";
        for (int i = 0; i < list->numCameras; ++i) {
            const char* id = list->cameraIds[i];
            ACameraMetadata* metadata = nullptr;
            if (ACameraManager_getCameraCharacteristics(manager_, id, &metadata) != ACAMERA_OK || !metadata) {
                continue;
            }
            ACameraMetadata_const_entry facingEntry{};
            int facing = -1;
            if (ACameraMetadata_getConstEntry(metadata, ACAMERA_LENS_FACING, &facingEntry) == ACAMERA_OK &&
                facingEntry.count > 0) {
                facing = facingEntry.data.u8[0];
            }
            seen << " id=" << id << " facing=" << facing;
            const bool matches = front
                    ? facing == ACAMERA_LENS_FACING_FRONT
                    : facing == ACAMERA_LENS_FACING_BACK;
            if (matches && outId.empty()) {
                outId = id;
                ACameraMetadata_const_entry orientationEntry{};
                if (ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_ORIENTATION,
                                                   &orientationEntry) == ACAMERA_OK &&
                    orientationEntry.count > 0) {
                    sensorOrientation_ = orientationEntry.data.i32[0];
                }
            }
            ACameraMetadata_free(metadata);
        }
        ACameraManager_deleteCameraIdList(list);
        return seen.str();
    }


    void chooseSupportedYuvSize(const std::string& cameraId, int requestedWidth, int requestedHeight) {
        ACameraMetadata* metadata = nullptr;
        if (ACameraManager_getCameraCharacteristics(manager_, cameraId.c_str(), &metadata) != ACAMERA_OK || !metadata) {
            width_ = requestedWidth;
            height_ = requestedHeight;
            return;
        }

        ACameraMetadata_const_entry configs{};
        if (ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &configs) != ACAMERA_OK ||
            configs.count < 4) {
            ACameraMetadata_free(metadata);
            width_ = requestedWidth;
            height_ = requestedHeight;
            return;
        }

        int exactW = 0;
        int exactH = 0;
        int bestSameAspectW = 0;
        int bestSameAspectH = 0;
        int64_t bestSameAspectArea = -1;
        std::ostringstream sizes;
        sizes << "YUV sizes:";

        auto aspectError = [&](int w, int h) -> int64_t {
            return std::llabs(static_cast<int64_t>(w) * requestedHeight -
                              static_cast<int64_t>(h) * requestedWidth);
        };
        auto isSameAspect = [&](int w, int h) -> bool {
            return aspectError(w, h) == 0;
        };
        for (uint32_t i = 0; i + 3 < configs.count; i += 4) {
            const int32_t format = configs.data.i32[i + 0];
            const int32_t w = configs.data.i32[i + 1];
            const int32_t h = configs.data.i32[i + 2];
            const int32_t input = configs.data.i32[i + 3];
            if (input != 0 || format != AIMAGE_FORMAT_YUV_420_888 || w <= 0 || h <= 0) continue;
            sizes << ' ' << w << 'x' << h;

            const int64_t area = static_cast<int64_t>(w) * h;
            if (w == requestedWidth && h == requestedHeight) {
                exactW = w;
                exactH = h;
            }
            if (w <= requestedWidth && h <= requestedHeight && isSameAspect(w, h) && area > bestSameAspectArea) {
                bestSameAspectArea = area;
                bestSameAspectW = w;
                bestSameAspectH = h;
            }
        }

        if (exactW > 0) {
            width_ = exactW;
            height_ = exactH;
        } else if (bestSameAspectW > 0) {
            width_ = bestSameAspectW;
            height_ = bestSameAspectH;
        } else {
            width_ = requestedWidth;
            height_ = requestedHeight;
        }
        LOGI("Requested YUV %dx%d, selected %dx%d fixed-aspect max-first. %s",
             requestedWidth, requestedHeight, width_, height_, sizes.str().c_str());
        ACameraMetadata_free(metadata);
    }

    void configureBestFpsRange(const std::string& cameraId) {
        ACameraMetadata* metadata = nullptr;
        if (ACameraManager_getCameraCharacteristics(manager_, cameraId.c_str(), &metadata) != ACAMERA_OK ||
            !metadata) {
            LOGI("AE FPS ranges unavailable; keeping camera default FPS");
            return;
        }

        ACameraMetadata_const_entry ranges{};
        if (ACameraMetadata_getConstEntry(metadata, ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES,
                                          &ranges) != ACAMERA_OK ||
            ranges.count < 2) {
            LOGI("AE FPS ranges missing; keeping camera default FPS");
            ACameraMetadata_free(metadata);
            return;
        }

        int bestMin = 0;
        int bestMax = 0;
        int bestScore = INT32_MIN;
        std::ostringstream seen;
        seen << "AE FPS ranges:";

        for (uint32_t i = 0; i + 1 < ranges.count; i += 2) {
            const int minFps = ranges.data.i32[i + 0];
            const int maxFps = ranges.data.i32[i + 1];
            seen << " " << minFps << "-" << maxFps;

            int score = INT32_MIN;
            if (minFps == 60 && maxFps == 60) {
                score = 60000;
            } else if (maxFps == 60 && minFps <= 60) {
                // Prefer ranges like 30-60, then 15-60.
                score = 50000 + minFps;
            } else if (minFps <= 60 && maxFps > 60) {
                // Last-choice 60-capable variable ranges, e.g. 30-120.
                score = 40000 - (maxFps - 60);
            } else if (minFps == 30 && maxFps == 30) {
                score = 30000;
            } else if (maxFps == 30 && minFps <= 30) {
                score = 25000 + minFps;
            } else if (maxFps < 30) {
                score = 10000 + maxFps;
            } else {
                score = 20000 + std::min(maxFps, 59);
            }

            if (score > bestScore) {
                bestScore = score;
                bestMin = minFps;
                bestMax = maxFps;
            }
        }

        if (bestMin > 0 && bestMax > 0) {
            const int32_t fpsRange[2] = {bestMin, bestMax};
            const camera_status_t status = ACaptureRequest_setEntry_i32(
                    request_,
                    ACAMERA_CONTROL_AE_TARGET_FPS_RANGE,
                    2,
                    fpsRange);
            if (status == ACAMERA_OK) {
                LOGI("%s | selected AE FPS %d-%d", seen.str().c_str(), bestMin, bestMax);
            } else {
                LOGE("%s | failed to set AE FPS %d-%d status=%d; keeping camera default FPS",
                     seen.str().c_str(), bestMin, bestMax, status);
            }
        } else {
            LOGI("%s | no usable range; keeping camera default FPS", seen.str().c_str());
        }

        ACameraMetadata_free(metadata);
    }

    std::string fail(const std::string& message) {
        LOGE("%s", message.c_str());
        stop();
        return "Error: " + message;
    }

    static void onDisconnected(void* context, ACameraDevice*) {
        static_cast<NativeYuvCamera*>(context)->running_ = false;
        LOGI("Camera disconnected");
    }

    static void onError(void* context, ACameraDevice*, int error) {
        static_cast<NativeYuvCamera*>(context)->running_ = false;
        LOGE("Camera error %d", error);
    }

    static void onSessionClosed(void*, ACameraCaptureSession*) {}
    static void onSessionReady(void*, ACameraCaptureSession*) {}
    static void onSessionActive(void*, ACameraCaptureSession*) {}

    static void onImageAvailable(void* context, AImageReader* reader) {
        auto* self = static_cast<NativeYuvCamera*>(context);
        if (!self || !self->running_) return;
        AImage* image = nullptr;
        media_status_t status = AImageReader_acquireLatestImage(reader, &image);
        if (status != AMEDIA_OK || !image) return;

        int64_t frameTimestampNs = 0;
        AImage_getTimestamp(image, &frameTimestampNs);

        int32_t width = 0;
        int32_t height = 0;
        AImage_getWidth(image, &width);
        AImage_getHeight(image, &height);

        if (width == self->width_ && height == self->height_) {
            const int planeW[3] = {width, width / 2, width / 2};
            const int planeH[3] = {height, height / 2, height / 2};
            bool valid = true;
            for (int p = 0; p < 3; ++p) {
                uint8_t* data = nullptr;
                int length = 0, rowStride = 0, pixelStride = 1;
                AImage_getPlaneData(image, p, &data, &length);
                AImage_getPlaneRowStride(image, p, &rowStride);
                AImage_getPlanePixelStride(image, p, &pixelStride);
                if (!data) { valid = false; break; }
                self->planes_[p].resize(static_cast<size_t>(planeW[p]) * planeH[p]);
                for (int row = 0; row < planeH[p]; ++row) {
                    const uint8_t* src = data + static_cast<size_t>(row) * rowStride;
                    uint8_t* dst = self->planes_[p].data() + static_cast<size_t>(row) * planeW[p];
                    if (pixelStride == 1) std::memcpy(dst, src, planeW[p]);
                    else for (int x = 0; x < planeW[p]; ++x) dst[x] = src[x * pixelStride];
                }
            }
            if (valid) {
                self->logFrameTiming(frameTimestampNs, width, height);
                CameraTransformation transform = camera_transform_evaluate();
                self->submitPreprocessedFrame(width, height, transform);
            }
        }
        AImage_delete(image);
    }

    void submitPreprocessedFrame(int32_t width, int32_t height, const CameraTransformation& transform) {
        const int64_t totalStartNs = nowNs();
        const int64_t convertStartNs = nowNs();
        if constexpr (kCameraPipelineFormat == CameraPipelineFormat::Yuv444) {
            const size_t pixelCount = static_cast<size_t>(width) * height;
            y444_.resize(pixelCount);
            u444_.resize(pixelCount);
            v444_.resize(pixelCount);
            std::memcpy(y444_.data(), planes_[0].data(), pixelCount);
            for (int y = 0; y < height; ++y) {
                const int chromaY = y >> 1;
                for (int x = 0; x < width; ++x) {
                    const int chromaX = x >> 1;
                    const size_t src = static_cast<size_t>(chromaY) * (width / 2) + chromaX;
                    const size_t dst = static_cast<size_t>(y) * width + x;
                    u444_[dst] = planes_[1][src];
                    v444_[dst] = planes_[2][src];
                }
            }
        }
        const int64_t convertEndNs = nowNs();

        CameraPreprocessGpuTiming gpuTiming{};
        const bool passThroughBecauseTargetIsLarger =
                kCameraDesiredOutputWidth >= width || kCameraDesiredOutputHeight >= height;
        const int processedWidth = passThroughBecauseTargetIsLarger ? width : kCameraDesiredOutputWidth;
        const int processedHeight = passThroughBecauseTargetIsLarger ? height : kCameraDesiredOutputHeight;
        const int64_t renderStartNs = nowNs();
        if constexpr (kCameraPipelineFormat == CameraPipelineFormat::Yuv420) {
            vulkanSubmitPreprocessedYuv420(planes_[0].data(), planes_[1].data(), planes_[2].data(),
                                           width, height,
                                           kCameraDesiredOutputWidth, kCameraDesiredOutputHeight,
                                           transform.rotation, transform.mirror);
        } else {
            vulkanSubmitPreprocessedYuv444(y444_.data(), u444_.data(), v444_.data(),
                                           width, height,
                                           kCameraDesiredOutputWidth, kCameraDesiredOutputHeight,
                                           transform.rotation, transform.mirror);
        }
        const int64_t renderEndNs = nowNs();
        logPreprocessTiming(convertEndNs - convertStartNs,
                            0,
                            renderEndNs - renderStartNs,
                            renderEndNs - totalStartNs,
                            width, height, processedWidth, processedHeight,
                            gpuTiming);
    }

    static int64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void logFrameTiming(int64_t frameTimestampNs, int32_t width, int32_t height) {
        const int64_t clockNs = nowNs();
        ++totalFrames_;
        if (fpsWindowStartNs_ <= 0) {
            fpsWindowStartNs_ = clockNs;
            fpsWindowFrames_ = 0;
        }
        ++fpsWindowFrames_;

        if (totalFrames_ == 1) {
            LOGI("first frame timestamp=%lld ns | %dx%d | camera=%s",
                 static_cast<long long>(frameTimestampNs),
                 width,
                 height,
                 useFront_ ? "front" : "back");
        }

        const int64_t elapsedNs = clockNs - fpsWindowStartNs_;
        if (elapsedNs >= 1'000'000'000LL) {
            const double fps = static_cast<double>(fpsWindowFrames_) *
                               1'000'000'000.0 / static_cast<double>(elapsedNs);
            LOGI("frame timestamp=%lld ns | fps=%.2f | frames=%llu | %dx%d | camera=%s",
                 static_cast<long long>(frameTimestampNs),
                 fps,
                 static_cast<unsigned long long>(totalFrames_),
                 width,
                 height,
                 useFront_ ? "front" : "back");
            fpsWindowStartNs_ = clockNs;
            fpsWindowFrames_ = 0;
        }
    }

    void logPreprocessTiming(int64_t convertNs,
                             int64_t pipelineNs,
                             int64_t renderNs,
                             int64_t totalNs,
                             int32_t inWidth,
                             int32_t inHeight,
                             int32_t outWidth,
                             int32_t outHeight,
                             const CameraPreprocessGpuTiming& gpuTiming) {
        const int64_t clockNs = nowNs();
        if (preprocessTimingWindowStartNs_ <= 0) {
            preprocessTimingWindowStartNs_ = clockNs;
        }

        ++preprocessTimingFrames_;
        preprocessConvertTotalNs_ += convertNs;
        preprocessPipelineTotalNs_ += pipelineNs;
        preprocessRenderTotalNs_ += renderNs;
        preprocessTotalNs_ += totalNs;
        preprocessGpuCommandTotalMs_ += gpuTiming.commandMs;
        preprocessGpuSobelTotalMs_ += gpuTiming.sobelMs;
        preprocessGpuMedianTotalMs_ += gpuTiming.medianMs;
        preprocessGpuFinalCopyTotalMs_ += gpuTiming.finalCopyMs;
        preprocessGpuCpuUploadTotalMs_ += gpuTiming.cpuUploadMapCopyMs;
        preprocessGpuCpuRecordTotalMs_ += gpuTiming.cpuCommandRecordMs;
        preprocessGpuCpuSubmitTotalMs_ += gpuTiming.cpuQueueSubmitMs;
        preprocessGpuCpuWaitTotalMs_ += gpuTiming.cpuFenceWaitMs;
        preprocessGpuCpuQueryTotalMs_ += gpuTiming.cpuQueryReadMs;
        preprocessGpuCpuOutputTotalMs_ += gpuTiming.cpuOutputMapCopyMs;
        preprocessGpuWallInternalTotalMs_ += gpuTiming.wallMs;

        const int64_t elapsedNs = clockNs - preprocessTimingWindowStartNs_;
        if (elapsedNs >= 1'000'000'000LL) {
            const double frames = static_cast<double>(preprocessTimingFrames_);
            if constexpr (kCameraPipelineFormat == CameraPipelineFormat::Yuv420) {
                LOGI("preprocess timing avg over %llu frames | pipeline_config=YUV420 | CPU_YUV420_prepare=%.2f ms | integrated_vulkan_420_preprocess_render=%.2f ms | total=%.2f ms | no CPU output readback | %dx%d -> %dx%d",
                     static_cast<unsigned long long>(preprocessTimingFrames_),
                     preprocessConvertTotalNs_ / frames / 1'000'000.0,
                     preprocessRenderTotalNs_ / frames / 1'000'000.0,
                     preprocessTotalNs_ / frames / 1'000'000.0,
                     inWidth,
                     inHeight,
                     outWidth,
                     outHeight);
            } else {
                LOGI("preprocess timing avg over %llu frames | pipeline_config=YUV444 | CPU_YUV420_to_YUV444=%.2f ms | integrated_vulkan_444_preprocess_render=%.2f ms | total=%.2f ms | no CPU output readback | %dx%d -> %dx%d",
                     static_cast<unsigned long long>(preprocessTimingFrames_),
                     preprocessConvertTotalNs_ / frames / 1'000'000.0,
                     preprocessRenderTotalNs_ / frames / 1'000'000.0,
                     preprocessTotalNs_ / frames / 1'000'000.0,
                     inWidth,
                     inHeight,
                     outWidth,
                     outHeight);
            }

            preprocessTimingWindowStartNs_ = clockNs;
            preprocessTimingFrames_ = 0;
            preprocessConvertTotalNs_ = 0;
            preprocessPipelineTotalNs_ = 0;
            preprocessRenderTotalNs_ = 0;
            preprocessTotalNs_ = 0;
            preprocessGpuCommandTotalMs_ = 0.0;
            preprocessGpuSobelTotalMs_ = 0.0;
            preprocessGpuMedianTotalMs_ = 0.0;
            preprocessGpuFinalCopyTotalMs_ = 0.0;
            preprocessGpuCpuUploadTotalMs_ = 0.0;
            preprocessGpuCpuRecordTotalMs_ = 0.0;
            preprocessGpuCpuSubmitTotalMs_ = 0.0;
            preprocessGpuCpuWaitTotalMs_ = 0.0;
            preprocessGpuCpuQueryTotalMs_ = 0.0;
            preprocessGpuCpuOutputTotalMs_ = 0.0;
            preprocessGpuWallInternalTotalMs_ = 0.0;
        }
    }

    bool useFront_ = false;
    bool running_ = false;
    int width_ = 1280;
    int height_ = 720;
    int sensorOrientation_ = 0;
    uint64_t totalFrames_ = 0;
    uint32_t fpsWindowFrames_ = 0;
    int64_t fpsWindowStartNs_ = 0;
    uint64_t preprocessTimingFrames_ = 0;
    int64_t preprocessTimingWindowStartNs_ = 0;
    int64_t preprocessConvertTotalNs_ = 0;
    int64_t preprocessPipelineTotalNs_ = 0;
    int64_t preprocessRenderTotalNs_ = 0;
    int64_t preprocessTotalNs_ = 0;
    double preprocessGpuCommandTotalMs_ = 0.0;
    double preprocessGpuSobelTotalMs_ = 0.0;
    double preprocessGpuMedianTotalMs_ = 0.0;
    double preprocessGpuFinalCopyTotalMs_ = 0.0;
    double preprocessGpuCpuUploadTotalMs_ = 0.0;
    double preprocessGpuCpuRecordTotalMs_ = 0.0;
    double preprocessGpuCpuSubmitTotalMs_ = 0.0;
    double preprocessGpuCpuWaitTotalMs_ = 0.0;
    double preprocessGpuCpuQueryTotalMs_ = 0.0;
    double preprocessGpuCpuOutputTotalMs_ = 0.0;
    double preprocessGpuWallInternalTotalMs_ = 0.0;
    std::string cameraId_;
    std::array<std::vector<uint8_t>, 3> planes_;
    std::vector<uint8_t> y444_;
    std::vector<uint8_t> u444_;
    std::vector<uint8_t> v444_;
    std::vector<uint8_t> processedY444_;
    std::vector<uint8_t> processedU444_;
    std::vector<uint8_t> processedV444_;

    ACameraManager* manager_ = nullptr;
    ACameraDevice* device_ = nullptr;
    AImageReader* reader_ = nullptr;
    ANativeWindow* readerWindow_ = nullptr;
    ACaptureSessionOutputContainer* outputs_ = nullptr;
    ACaptureSessionOutput* readerOutput_ = nullptr;
    ACameraOutputTarget* target_ = nullptr;
    ACaptureRequest* request_ = nullptr;
    ACameraCaptureSession* session_ = nullptr;
    ACameraDevice_StateCallbacks cameraCallbacks_{};
    ACameraCaptureSession_stateCallbacks sessionCallbacks_{};
};

std::mutex gMutex;
std::unique_ptr<NativeYuvCamera> gCamera;

NativeYuvCamera* camera() {
    if (!gCamera) gCamera = std::make_unique<NativeYuvCamera>();
    return gCamera.get();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeStart(
        JNIEnv* env, jclass, jboolean front, jint width, jint height) {
    std::lock_guard<std::mutex> lock(gMutex);
    std::string result = camera()->start(front == JNI_TRUE, width, height);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeStop(
        JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (gCamera) gCamera->stop();
}

extern "C" JNIEXPORT jstring JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeSetSurface(
        JNIEnv* env, jclass, jobject surface, jobject assetManager) {
    if (!surface) {
        cameraPreprocessGpuDestroy();
        vulkanDestroy();
        return env->NewStringUTF("Vulkan surface released");
    }
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    AAssetManager* assets = AAssetManager_fromJava(env, assetManager);
    if (!cameraPreprocessGpuLoadShaders(assets)) {
        ANativeWindow_release(window);
        return env->NewStringUTF("Error: failed to load camera preprocessing Vulkan shaders");
    }
    std::string result = vulkanSetWindow(window, assets);
    ANativeWindow_release(window);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeSetDisplayRotation(
        JNIEnv*, jclass, jint rotationDegrees) {
    camera_transform_set_display_rotation(rotationDegrees);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeCaptureLatestFilteredArgb(
        JNIEnv* env, jclass, jintArray dimensions) {
    std::vector<uint32_t> pixels;
    int width = 0;
    int height = 0;
    if (!vulkanCopyLatestFilteredArgb(pixels, width, height) || pixels.empty()) {
        return nullptr;
    }
    if (dimensions && env->GetArrayLength(dimensions) >= 2) {
        const jint values[2] = {width, height};
        env->SetIntArrayRegion(dimensions, 0, 2, values);
    }
    jintArray result = env->NewIntArray(static_cast<jsize>(pixels.size()));
    if (!result) return nullptr;
    env->SetIntArrayRegion(result, 0, static_cast<jsize>(pixels.size()),
                           reinterpret_cast<const jint*>(pixels.data()));
    return result;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_app_builderx_ogfa_camerapipelinetest_CameraActivity_nativeCaptureFilteredYuv420Ring(
        JNIEnv* env, jclass, jintArray info) {
    std::array<std::vector<uint8_t>, 3> downsampledFrames;
    std::array<std::vector<uint8_t>, 3> filteredFrames;
    int width = 0;
    int height = 0;
    int chromaWidth = 0;
    int chromaHeight = 0;
    int paddedWidth = 0;
    int paddedHeight = 0;
    int paddedChromaWidth = 0;
    int paddedChromaHeight = 0;
    if (!vulkanCopyYuv420Rings(downsampledFrames, filteredFrames, width, height,
                               chromaWidth, chromaHeight, paddedWidth, paddedHeight,
                               paddedChromaWidth, paddedChromaHeight)) {
        return nullptr;
    }
    if (info && env->GetArrayLength(info) >= 9) {
        const jint values[9] = {
                width,
                height,
                chromaWidth,
                chromaHeight,
                paddedWidth,
                paddedHeight,
                paddedChromaWidth,
                paddedChromaHeight,
                static_cast<jint>(filteredFrames[0].size())
        };
        env->SetIntArrayRegion(info, 0, 9, values);
    }
    jclass byteArrayClass = env->FindClass("[B");
    jobjectArray result = env->NewObjectArray(6, byteArrayClass, nullptr);
    if (!result) return nullptr;
    for (jsize i = 0; i < 6; ++i) {
        const std::vector<uint8_t>& source = i < 3 ? downsampledFrames[i] : filteredFrames[i - 3];
        jbyteArray frame = env->NewByteArray(static_cast<jsize>(source.size()));
        if (!frame) return nullptr;
        env->SetByteArrayRegion(frame, 0, static_cast<jsize>(source.size()),
                                reinterpret_cast<const jbyte*>(source.data()));
        env->SetObjectArrayElement(result, i, frame);
        env->DeleteLocalRef(frame);
    }
    return result;
}
