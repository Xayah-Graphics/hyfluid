#ifndef HYFLUID_TRAIN_CONFIG_H
#define HYFLUID_TRAIN_CONFIG_H

#include <cstdint>

namespace hyfluid::train::config {
    inline constexpr std::uint32_t initial_rays_per_batch = 1u << 12u;
    inline constexpr std::uint32_t nerf_grid_size         = 128u;
    inline constexpr std::uint32_t nerf_grid_cells        = nerf_grid_size * nerf_grid_size * nerf_grid_size;
    inline constexpr std::uint32_t nerf_steps             = 1024u;
    inline constexpr std::uint32_t max_samples            = initial_rays_per_batch * nerf_steps;
    inline constexpr std::uint32_t sample_coord_floats    = 5u;
    inline constexpr std::uint32_t ray_floats             = 6u;
    inline constexpr std::uint32_t threads_per_block      = 128u;
    inline constexpr std::uint32_t max_random_samples_per_ray = 16u;
    inline constexpr float min_cone_stepsize              = 1.73205080757f / static_cast<float>(nerf_steps);
    inline constexpr bool snap_to_pixel_centers           = true;

    inline constexpr std::uint64_t train_seed           = 1337u;
    inline constexpr std::uint64_t pcg32_default_state  = 0x853c49e6748fea9bULL;
    inline constexpr std::uint64_t pcg32_default_stream = 0xda3e39cb94b95bdbULL;
    inline constexpr std::uint64_t pcg32_mult           = 0x5851f42d4c957f2dULL;

    static_assert((nerf_grid_size & (nerf_grid_size - 1u)) == 0u);
    static_assert(nerf_grid_cells % 8u == 0u);
    static_assert(sample_coord_floats == 5u);
    static_assert(ray_floats == 6u);
    static_assert(max_samples != 0u);
} // namespace hyfluid::train::config

#endif // HYFLUID_TRAIN_CONFIG_H
