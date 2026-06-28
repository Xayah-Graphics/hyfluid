#ifndef HYFLUID_TRAIN_H
#define HYFLUID_TRAIN_H

#include "hyfluid.train.config.h"
#include <cstdint>
#include <type_traits>

namespace hyfluid::cuda {
    // Device memory.
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

    // Dataset.
    void upload_dataset(const std::uint8_t* pixels, std::size_t pixel_bytes, const float* camera, std::size_t camera_count, const float* intrinsics, std::size_t intrinsics_count, const float* times, std::size_t time_count, const std::uint32_t* view_indices, std::size_t view_index_count, const std::uint32_t* time_indices, std::size_t time_index_count, const std::uint32_t* frame_indices, std::size_t frame_index_count, const std::uint8_t*& out_pixels, const float*& out_camera, const float*& out_intrinsics, const float*& out_times, const std::uint32_t*& out_view_indices, const std::uint32_t*& out_time_indices, const std::uint32_t*& out_frame_indices);

    // Sampler.
    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy, std::uint32_t*& out_occupancy_grid_occupied_count);
    void set_occupancy_grid_full(std::uint8_t* occupancy, std::uint32_t* occupancy_grid_occupied_count);
    void sample_training_batch(const float* camera, const float* intrinsics, const float* times, const std::uint32_t* frame_indices, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);

    // Host readback.
    void read_counter(const std::uint32_t* counter, std::uint32_t& out_value);
} // namespace hyfluid::cuda

#endif // HYFLUID_TRAIN_H
