#ifndef HYFLUID_TRAIN_H
#define HYFLUID_TRAIN_H

#include "hyfluid.train.config.h"
#include <cstdint>
#include <type_traits>

namespace hyfluid::cuda {
    void free_device_buffers(void** pointers, std::size_t count) noexcept;

    template <typename... Pointers>
        requires ((std::is_pointer_v<Pointers> && ...))
    void free_device_buffers(Pointers&... pointers) noexcept {
        if constexpr (sizeof...(Pointers) > 0u) {
            void* raw[] = {const_cast<void*>(static_cast<const void*>(pointers))...};
            free_device_buffers(raw, sizeof...(Pointers));
            ((pointers = nullptr), ...);
        }
    }

    void upload_dataset(const std::uint8_t* pixels, std::size_t pixel_bytes, const float* camera, std::size_t camera_count, const float* intrinsics, std::size_t intrinsics_count, const float* times, std::size_t time_count, const std::uint32_t* view_indices, std::size_t view_index_count, const std::uint32_t* time_indices, std::size_t time_index_count, const std::uint32_t* frame_indices, std::size_t frame_index_count, const std::uint8_t*& out_pixels, const float*& out_camera, const float*& out_intrinsics, const float*& out_times, const std::uint32_t*& out_view_indices, const std::uint32_t*& out_time_indices, const std::uint32_t*& out_frame_indices);
} // namespace hyfluid::cuda

#endif // HYFLUID_TRAIN_H
