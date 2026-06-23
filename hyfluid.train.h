#ifndef HYFLUID_TRAIN_H
#define HYFLUID_TRAIN_H

#include "hyfluid.train.config.h"
#include <cstdint>

namespace hyfluid::cuda {
    void free_dataset(std::uint8_t*& pixels, float*& camera) noexcept;
    void upload_dataset(const std::uint8_t* pixels, std::size_t pixels_bytes, const float* camera, std::size_t camera_count, std::uint8_t*& out_pixels, float*& out_camera);
} // namespace hyfluid::cuda

#endif // HYFLUID_TRAIN_H
