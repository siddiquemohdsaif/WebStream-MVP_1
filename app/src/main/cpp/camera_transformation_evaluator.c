#include "camera_transformation_evaluator.h"

#include <android/log.h>
#include <pthread.h>
#include <unistd.h>

#define TAG "CameraTransform"

static pthread_mutex_t state_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t logger_thread;
static int sensor_orientation = 0;
static int display_rotation = 0;
static bool is_front_camera = false;
static bool logger_running = false;
static bool logger_created = false;

static int normalize_rotation(int degrees) {
    degrees %= 360;
    if (degrees < 0) degrees += 360;

    /* Inputs should already be quarter turns; this makes the API defensive. */
    return ((degrees + 45) / 90 * 90) % 360;
}

void camera_transform_set_camera(int orientation, bool front_camera) {
    pthread_mutex_lock(&state_mutex);
    sensor_orientation = normalize_rotation(orientation);
    is_front_camera = front_camera;
    pthread_mutex_unlock(&state_mutex);
}

void camera_transform_set_display_rotation(int rotation) {
    pthread_mutex_lock(&state_mutex);
    display_rotation = normalize_rotation(rotation);
    pthread_mutex_unlock(&state_mutex);
}

CameraTransformation camera_transform_evaluate(void) {
    CameraTransformation result;
    pthread_mutex_lock(&state_mutex);
    /*
     * This metadata describes the raw camera buffer only. Display rotation is
     * handled independently by the local/remote Vulkan presentation layer.
     */
    result.rotation = (uint16_t)sensor_orientation;
    result.mirror = is_front_camera;
    pthread_mutex_unlock(&state_mutex);
    return result;
}

static void* logger_main(void* unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&state_mutex);
        bool running = logger_running;
        int sensor = sensor_orientation;
        int display = display_rotation;
        bool front = is_front_camera;
        pthread_mutex_unlock(&state_mutex);
        if (!running) break;

        CameraTransformation value = camera_transform_evaluate();
        __android_log_print(ANDROID_LOG_INFO, TAG,
                            "rotation=%u mirror=%s | sensor=%d display=%d camera=%s",
                            value.rotation, value.mirror ? "true" : "false",
                            sensor, display, front ? "front" : "back");
        sleep(1);
    }
    return NULL;
}

void camera_transform_start_logging(void) {
    pthread_mutex_lock(&state_mutex);
    if (logger_running) {
        pthread_mutex_unlock(&state_mutex);
        return;
    }
    logger_running = true;
    logger_created = pthread_create(&logger_thread, NULL, logger_main, NULL) == 0;
    if (!logger_created) logger_running = false;
    pthread_mutex_unlock(&state_mutex);
}

void camera_transform_stop_logging(void) {
    pthread_mutex_lock(&state_mutex);
    bool join = logger_created;
    logger_running = false;
    logger_created = false;
    pthread_mutex_unlock(&state_mutex);
    if (join) pthread_join(logger_thread, NULL);
}
