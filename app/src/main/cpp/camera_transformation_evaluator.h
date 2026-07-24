#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CameraTransformation {
    uint16_t rotation;
    bool mirror;
} CameraTransformation;

void camera_transform_set_camera(int sensor_orientation, bool front_camera);
void camera_transform_set_display_rotation(int display_rotation);
CameraTransformation camera_transform_evaluate(void);
void camera_transform_start_logging(void);
void camera_transform_stop_logging(void);

#ifdef __cplusplus
}
#endif
