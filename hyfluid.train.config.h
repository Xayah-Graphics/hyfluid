#ifndef HYFLUID_TRAIN_CONFIG_H
#define HYFLUID_TRAIN_CONFIG_H

#include <array>
#include <cstdint>

namespace hyfluid::train::config {
    inline constexpr std::uint32_t hash4_level_count        = 16u;
    inline constexpr std::uint32_t hash4_features_per_level = 2u;
    inline constexpr std::uint32_t hash4_output_width       = hash4_level_count * hash4_features_per_level;
    inline constexpr std::uint32_t hash4_max_entries        = 1u << 19u;
    inline constexpr std::array hash4_resolutions           = {16u, 21u, 26u, 32u, 41u, 51u, 65u, 81u, 102u, 129u, 162u, 204u, 257u, 323u, 407u, 512u};

    constexpr std::uint32_t ceil_to_uint(const float value) {
        const auto truncated = static_cast<std::uint32_t>(value);
        return static_cast<float>(truncated) == value ? truncated : truncated + 1u;
    }

    inline constexpr std::uint32_t mlp_width            = 64u;
    inline constexpr std::uint32_t mlp_input_width      = hash4_output_width;
    inline constexpr std::uint32_t network_output_width = 1u;

    inline constexpr std::uint32_t network_batch_size            = 1u << 18u;
    inline constexpr std::uint32_t evaluation_network_batch_size = 1u << 21u;
    inline constexpr std::uint32_t network_buffer_capacity       = evaluation_network_batch_size > network_batch_size ? evaluation_network_batch_size : network_batch_size;
    inline constexpr std::uint32_t network_batch_granularity     = 128u;
    inline constexpr std::uint32_t initial_rays_per_batch        = 1024u;
    inline constexpr std::uint32_t nerf_grid_size                = 128u;
    inline constexpr std::uint32_t nerf_grid_cells               = nerf_grid_size * nerf_grid_size * nerf_grid_size;
    inline constexpr std::uint32_t nerf_grid_bitfield_bytes      = nerf_grid_cells / 8u;
    inline constexpr float nerf_min_optical_thickness            = 0.01f;
    inline constexpr float unit_aabb_diagonal                    = 1.73205080757f;
    inline constexpr std::uint32_t training_ray_steps            = 192u;
    inline constexpr std::uint32_t max_hash_resolution           = hash4_resolutions[hash4_level_count - 1u];
    inline constexpr std::uint32_t evaluation_ray_step_factor    = 2u;
    inline constexpr std::uint32_t evaluation_ray_steps          = ceil_to_uint(unit_aabb_diagonal * static_cast<float>(max_hash_resolution * evaluation_ray_step_factor));
    inline constexpr std::uint32_t max_samples                   = network_batch_size;
    inline constexpr std::uint32_t max_rays_per_batch            = (network_batch_size / training_ray_steps / network_batch_granularity) * network_batch_granularity;
    inline constexpr std::uint32_t sample_coord_floats           = 5u;
    inline constexpr std::uint32_t ray_floats                    = 6u;
    inline constexpr std::uint32_t evaluation_tile_pixels        = evaluation_network_batch_size / evaluation_ray_steps;
    inline constexpr std::uint32_t threads_per_block             = 128u;
    inline constexpr std::uint32_t grid_forward_threads       = 256u;
    inline constexpr std::uint32_t grid_backward_threads      = 256u;
    inline constexpr std::uint32_t max_random_samples_per_ray = 16u;
    inline constexpr std::uint32_t random_values_per_thread   = 4u;
    inline constexpr std::size_t cublaslt_workspace_bytes     = static_cast<std::size_t>(64u) * 1024u * 1024u;
    inline constexpr float training_ray_stepsize              = unit_aabb_diagonal / static_cast<float>(training_ray_steps);
    inline constexpr float evaluation_ray_stepsize            = unit_aabb_diagonal / static_cast<float>(evaluation_ray_steps);
    inline constexpr float transmittance_epsilon              = 1.0e-4f;

    inline constexpr std::uint32_t density_grid_warmup_passes              = 2u;
    inline constexpr std::uint32_t density_grid_skip_interval              = 16u;
    inline constexpr std::uint32_t density_grid_max_skip                   = 16u;
    inline constexpr std::uint32_t density_grid_warmup_samples             = nerf_grid_cells;
    inline constexpr std::uint32_t density_grid_steady_uniform_samples     = nerf_grid_cells / 16u;
    inline constexpr std::uint32_t density_grid_steady_nonuniform_samples  = nerf_grid_cells / 16u;
    inline constexpr float density_grid_decay                              = 0.95f;

    inline constexpr float optimizer_learning_rate = 5.0e-4f;
    inline constexpr float optimizer_beta1         = 0.9f;
    inline constexpr float optimizer_beta2         = 0.99f;
    inline constexpr float optimizer_epsilon       = 1.0e-15f;
    inline constexpr float optimizer_weight_decay  = 1.0e-6f;
    inline constexpr float optimizer_loss_scale    = 64.0f;

    inline constexpr std::uint64_t train_seed           = 1337u;
    inline constexpr std::uint64_t pcg32_default_state  = 0x853c49e6748fea9bULL;
    inline constexpr std::uint64_t pcg32_default_stream = 0xda3e39cb94b95bdbULL;
    inline constexpr std::uint64_t pcg32_mult           = 0x5851f42d4c957f2dULL;

    struct NetworkParameterLayout final {
        std::array<std::uint32_t, hash4_level_count + 1u> hash4_offsets = {};
        std::uint32_t mlp_input_weight_offset                           = 0u;
        std::uint32_t mlp_output_weight_offset                          = 0u;
        std::uint32_t global_rgb_offset                                 = 0u;
        std::uint32_t mlp_param_count                                   = 0u;
        std::uint32_t hash4_param_offset                                = 0u;
        std::uint32_t hash4_param_count                                 = 0u;
        std::uint32_t total_param_count                                 = 0u;
    };

    constexpr std::uint64_t round_up_8(const std::uint64_t value) {
        return ((value + 7ull) / 8ull) * 8ull;
    }

    constexpr std::uint32_t hash4_level_entries(const std::uint32_t resolution) {
        const std::uint64_t axis            = static_cast<std::uint64_t>(resolution) + 1ull;
        const std::uint64_t dense_entries   = axis * axis * axis * axis;
        const std::uint64_t rounded_entries = round_up_8(dense_entries);
        return static_cast<std::uint32_t>(rounded_entries < hash4_max_entries ? rounded_entries : hash4_max_entries);
    }

    constexpr NetworkParameterLayout make_network_parameter_layout() {
        NetworkParameterLayout layout = {};
        std::uint32_t hash4_cursor    = 0u;
        for (std::uint32_t level = 0u; level < hash4_level_count; ++level) {
            layout.hash4_offsets[level] = hash4_cursor;
            hash4_cursor += hash4_level_entries(hash4_resolutions[level]) * hash4_features_per_level;
        }
        layout.hash4_offsets[hash4_level_count] = hash4_cursor;

        std::uint32_t param_cursor     = 0u;
        layout.mlp_input_weight_offset = param_cursor;
        param_cursor += mlp_width * mlp_input_width;
        layout.mlp_output_weight_offset = param_cursor;
        param_cursor += network_output_width * mlp_width;
        layout.global_rgb_offset = param_cursor;
        param_cursor += 1u;
        layout.mlp_param_count    = param_cursor;
        layout.hash4_param_offset = param_cursor;
        layout.hash4_param_count  = hash4_cursor;
        param_cursor += layout.hash4_param_count;
        layout.total_param_count = param_cursor;
        return layout;
    }

    inline constexpr NetworkParameterLayout network_parameter_layout = make_network_parameter_layout();
    inline constexpr std::uint32_t hash4_offset_0                    = network_parameter_layout.hash4_offsets[0u];
    inline constexpr std::uint32_t hash4_offset_1                    = network_parameter_layout.hash4_offsets[1u];
    inline constexpr std::uint32_t hash4_offset_2                    = network_parameter_layout.hash4_offsets[2u];
    inline constexpr std::uint32_t hash4_offset_3                    = network_parameter_layout.hash4_offsets[3u];
    inline constexpr std::uint32_t hash4_offset_4                    = network_parameter_layout.hash4_offsets[4u];
    inline constexpr std::uint32_t hash4_offset_5                    = network_parameter_layout.hash4_offsets[5u];
    inline constexpr std::uint32_t hash4_offset_6                    = network_parameter_layout.hash4_offsets[6u];
    inline constexpr std::uint32_t hash4_offset_7                    = network_parameter_layout.hash4_offsets[7u];
    inline constexpr std::uint32_t hash4_offset_8                    = network_parameter_layout.hash4_offsets[8u];
    inline constexpr std::uint32_t hash4_offset_9                    = network_parameter_layout.hash4_offsets[9u];
    inline constexpr std::uint32_t hash4_offset_10                   = network_parameter_layout.hash4_offsets[10u];
    inline constexpr std::uint32_t hash4_offset_11                   = network_parameter_layout.hash4_offsets[11u];
    inline constexpr std::uint32_t hash4_offset_12                   = network_parameter_layout.hash4_offsets[12u];
    inline constexpr std::uint32_t hash4_offset_13                   = network_parameter_layout.hash4_offsets[13u];
    inline constexpr std::uint32_t hash4_offset_14                   = network_parameter_layout.hash4_offsets[14u];
    inline constexpr std::uint32_t hash4_offset_15                   = network_parameter_layout.hash4_offsets[15u];
    inline constexpr std::uint32_t hash4_offset_16                   = network_parameter_layout.hash4_offsets[16u];

    static_assert((nerf_grid_size & (nerf_grid_size - 1u)) == 0u);
    static_assert(nerf_grid_cells % 8u == 0u);
    static_assert(nerf_grid_bitfield_bytes * 8u == nerf_grid_cells);
    static_assert(hash4_level_count == 16u);
    static_assert(hash4_features_per_level == 2u);
    static_assert(hash4_output_width == 32u);
    static_assert(mlp_width == 64u);
    static_assert(network_output_width == 1u);
    static_assert(network_batch_size % network_batch_granularity == 0u);
    static_assert(sample_coord_floats == 5u);
    static_assert(ray_floats == 6u);
    static_assert(training_ray_steps != 0u);
    static_assert(max_hash_resolution == 512u);
    static_assert(evaluation_ray_steps == 1774u);
    static_assert(evaluation_network_batch_size >= network_batch_size);
    static_assert(evaluation_tile_pixels != 0u);
    static_assert(evaluation_tile_pixels * evaluation_ray_steps <= evaluation_network_batch_size);
    static_assert(network_buffer_capacity >= evaluation_network_batch_size);
    static_assert(network_buffer_capacity >= network_batch_size);
    static_assert(max_samples != 0u);
    static_assert(max_samples == network_batch_size);
    static_assert(max_rays_per_batch >= initial_rays_per_batch);
    static_assert(max_rays_per_batch % network_batch_granularity == 0u);
    static_assert(max_rays_per_batch * training_ray_steps <= network_batch_size);
    static_assert(density_grid_warmup_samples <= nerf_grid_cells);
    static_assert(density_grid_warmup_samples % network_batch_size == 0u);
    static_assert(density_grid_warmup_passes != 0u);
    static_assert(density_grid_steady_uniform_samples + density_grid_steady_nonuniform_samples <= max_samples);
    static_assert(network_parameter_layout.hash4_offsets[0u] == 0u);
    static_assert(network_parameter_layout.mlp_input_weight_offset == 0u);
    static_assert(network_parameter_layout.mlp_output_weight_offset == mlp_width * mlp_input_width);
    static_assert(network_parameter_layout.global_rgb_offset == network_parameter_layout.mlp_output_weight_offset + network_output_width * mlp_width);
    static_assert(network_parameter_layout.hash4_param_offset == network_parameter_layout.mlp_param_count);
    static_assert(network_parameter_layout.total_param_count == network_parameter_layout.hash4_param_offset + network_parameter_layout.hash4_param_count);
    static_assert(hash4_offset_16 == network_parameter_layout.hash4_param_count);
} // namespace hyfluid::train::config

#endif // HYFLUID_TRAIN_CONFIG_H
