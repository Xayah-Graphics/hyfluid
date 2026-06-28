#ifndef HYFLUID_INSPECTOR_H
#define HYFLUID_INSPECTOR_H

#include <cstddef>
#include <cstdint>

namespace hyfluid::cuda {
    void fill_sampler_visualization(std::uint32_t ray_count, std::uint32_t sample_count, const float* rays, const std::uint32_t* numsteps, const float* sample_coords, std::uint32_t time_count, std::uint32_t time_index, float point_radius, float ray_width, std::uint32_t width_mode, std::byte* point_instances, std::uint64_t point_byte_size, std::byte* segment_instances, std::uint64_t segment_byte_size, std::uint32_t& out_point_count, std::uint32_t& out_ray_count);
} // namespace hyfluid::cuda

#endif // HYFLUID_INSPECTOR_H
