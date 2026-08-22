// MediaPipe Hands C++ Implementation
// Minimal glue code (~100 lines) for hand landmark detection

#include "hand_tracking.h"
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>

using mediapipe::CalculatorGraph;
using mediapipe::ImageFrame;
using mediapipe::NormalizedLandmark;
using mediapipe::tasks::vision::hand_landmarker::HandLandmarker;
using mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult;

namespace {
    std::unique_ptr<HandLandmarker> g_landmarker = nullptr;
    HandResult g_last_result = {};
    float g_last_velocity = 0.0f;
    float g_last_processing_time_ms = 0.0f;
    
    // Previous wrist position for velocity calculation
    float g_prev_wrist_x = 0.0f;
    float g_prev_wrist_y = 0.0f;
    bool g_has_previous = false;
}

extern "C" {

int hand_tracking_init(void) {
    if (g_landmarker != nullptr) {
        return 0; // Already initialized
    }
    
    try {
        auto options = HandLandmarker::Options();
        options.base_options.model_asset_path = "hand_landmarker.task";
        options.running_mode = mediapipe::tasks::core::RunningMode::VIDEO;
        options.num_hands = 1;
        
        g_landmarker = std::make_unique<HandLandmarker>(
            HandLandmarker::CreateFromOptions(options));
        
        if (!g_landmarker) {
            return -1;
        }
        
        memset(&g_last_result, 0, sizeof(g_last_result));
        g_has_previous = false;
        
        return 0;
    } catch (...) {
        return -2;
    }
}

HandResult hand_tracking_process_frame(
    const uint8_t* y_plane,
    const uint8_t* u_plane,
    const uint8_t* v_plane,
    int width,
    int height,
    int y_row_stride,
    int uv_row_stride
) {
    HandResult result = {};
    result.is_present = false;
    
    if (!g_landmarker || !y_plane) {
        return result;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Convert YUV420 to RGB (simplified - in production use proper conversion)
        std::vector<uint8_t> rgb_data(width * height * 3);
        
        // Basic YUV to RGB conversion (YUV420 planar)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int y_idx = y * y_row_stride + x;
                int uv_y = y / 2;
                int uv_x = x / 2;
                int uv_idx = uv_y * uv_row_stride + uv_x;
                
                float Y = y_plane[y_idx] / 255.0f;
                float U = u_plane[uv_idx] / 255.0f - 0.5f;
                float V = v_plane[uv_idx] / 255.0f - 0.5f;
                
                float R = Y + 1.402f * V;
                float G = Y - 0.344136f * U - 0.714136f * V;
                float B = Y + 1.772f * U;
                
                int rgb_idx = (y * width + x) * 3;
                rgb_data[rgb_idx + 0] = static_cast<uint8_t>(std::clamp(R, 0.0f, 1.0f) * 255);
                rgb_data[rgb_idx + 1] = static_cast<uint8_t>(std::clamp(G, 0.0f, 1.0f) * 255);
                rgb_data[rgb_idx + 2] = static_cast<uint8_t>(std::clamp(B, 0.0f, 1.0f) * 255);
            }
        }
        
        // Create ImageFrame for MediaPipe
        auto image_frame = std::make_unique<ImageFrame>(
            ImageFrame::SRGB, width, height);
        std::memcpy(image_frame->MutablePixelData(), rgb_data.data(), 
                    rgb_data.size());
        
        // Run hand landmarker
        auto mp_image = mediapipe::MakePacket<std::unique_ptr<ImageFrame>>(
            std::move(image_frame));
        
        auto packet_map = g_landmarker->DetectForVideo(mp_image, 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        
        if (packet_map.empty()) {
            return result;
        }
        
        const auto& hand_results = packet_map["HAND_LANDMARKS"].Get<HandLandmarkerResult>();
        
        if (hand_results.hand_landmarks.empty()) {
            return result;
        }
        
        // Extract first hand's 21 landmarks
        const auto& landmarks = hand_results.hand_landmarks[0];
        
        for (int i = 0; i < 21 && i < landmarks.size(); ++i) {
            result.landmarks[i].x = landmarks[i].x();
            result.landmarks[i].y = landmarks[i].y();
            result.landmarks[i].z = landmarks[i].z();
        }
        
        result.confidence = hand_results.hand_world_landmarks.empty() ? 
            0.0f : 1.0f; // Simplified confidence
        result.is_present = true;
        
        // Calculate wrist velocity (landmark 0)
        if (g_has_previous) {
            float dx = result.landmarks[0].x - g_prev_wrist_x;
            float dy = result.landmarks[0].y - g_prev_wrist_y;
            g_last_velocity = std::sqrt(dx * dx + dy * dy) * 30.0f; // Scale by FPS
        }
        
        g_prev_wrist_x = result.landmarks[0].x;
        g_prev_wrist_y = result.landmarks[0].y;
        g_has_previous = true;
        
        g_last_result = result;
        
    } catch (...) {
        result.is_present = false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    g_last_processing_time_ms = std::chrono::duration<float, std::milli>(
        end - start).count();
    
    return result;
}

float hand_tracking_get_wrist_velocity(void) {
    return g_last_velocity;
}

bool hand_tracking_is_hand_present(void) {
    return g_last_result.is_present;
}

void hand_tracking_cleanup(void) {
    g_landmarker.reset();
    memset(&g_last_result, 0, sizeof(g_last_result));
    g_last_velocity = 0.0f;
    g_has_previous = false;
}

float hand_tracking_get_last_processing_time_ms(void) {
    return g_last_processing_time_ms;
}

} // extern "C"
