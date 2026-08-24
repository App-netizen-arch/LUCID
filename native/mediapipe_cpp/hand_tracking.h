#ifndef HAND_TRACKING_H
#define HAND_TRACKING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Hand landmark structure (21 landmarks x 3 coordinates)
typedef struct {
    float x;
    float y;
    float z;
} Landmark3D;

typedef struct {
    Landmark3D landmarks[21];
    float confidence;
    bool is_present;
} HandResult;

// Initialize MediaPipe Hands tracker
// Returns 0 on success, non-zero on failure
int hand_tracking_init(void);

// Process a single frame (YUV420 format from camera)
// width, height: frame dimensions
// y_plane, u_plane, v_plane: YUV data pointers
// y_row_stride, uv_row_stride: row strides
// Returns HandResult with 21 landmarks if hand detected
HandResult hand_tracking_process_frame(
    const uint8_t* y_plane,
    const uint8_t* u_plane,
    const uint8_t* v_plane,
    int width,
    int height,
    int y_row_stride,
    int uv_row_stride
);

// Get wrist velocity energy for segmentation
// Returns velocity magnitude in normalized units
float hand_tracking_get_wrist_velocity(void);

// Check if hand is present in current frame
bool hand_tracking_is_hand_present(void);

// Cleanup and release resources
void hand_tracking_cleanup(void);

// Get last processing time in milliseconds
float hand_tracking_get_last_processing_time_ms(void);

#ifdef __cplusplus
}
#endif

#endif // HAND_TRACKING_H
