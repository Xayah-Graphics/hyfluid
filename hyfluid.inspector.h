#ifndef HYFLUID_INSPECTOR_H
#define HYFLUID_INSPECTOR_H

#include <cstddef>
#include <cstdint>

namespace hyfluid::cuda {
    struct TrainingBatchDiagnostics final {
        float coord_min[3u] = {};
        float coord_max[3u] = {};
        float time_min = 0.0f;
        float time_max = 0.0f;
        float dt_metric_min = 0.0f;
        float dt_metric_mean = 0.0f;
        float dt_metric_max = 0.0f;
        float metric_per_field_unit_min = 0.0f;
        float metric_per_field_unit_mean = 0.0f;
        float metric_per_field_unit_max = 0.0f;
    };

    void read_float_value(const float* values, std::uint32_t index, float& out_value);
    void read_training_batch_diagnostics(const float* sample_coords, std::uint32_t sample_count, TrainingBatchDiagnostics& out_diagnostics);
    void sample_density_slice(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z, float time, const std::uint16_t* params, float* sample_coords, std::uint16_t* network_input, std::uint16_t* network_hidden, std::uint16_t* network_output, void* cublaslt_handle, std::uint8_t* cublaslt_workspace, float* output_density, float& out_density_min, float& out_density_max, float& out_density_mean, std::uint64_t& out_density_nonzero_count);
    void fill_sampler_visualization(std::uint32_t ray_count, std::uint32_t sample_count, const float* rays, const std::uint32_t* numsteps, const float* sample_coords, std::uint32_t time_count, std::uint32_t time_index, float point_radius, float ray_width, std::uint32_t width_mode, std::byte* point_instances, std::uint64_t point_byte_size, std::byte* segment_instances, std::uint64_t segment_byte_size, std::uint32_t& out_point_count, std::uint32_t& out_ray_count);
} // namespace hyfluid::cuda

#endif // HYFLUID_INSPECTOR_H
