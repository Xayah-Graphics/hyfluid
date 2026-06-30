module;
#include "hyfluid.inspector.h"

#include "hyfluid.train.h"
module hyfluid.inspector;
import std;
import hyfluid.train;

namespace hyfluid::inspector {
    static_assert(DensitySliceDimension == train::config::nerf_grid_size);

    Inspector::Inspector(const train::HyFluid& trainer) : trainer{std::addressof(trainer)} {}

    TrainingDomainView Inspector::training_domain_view() const {
        const OccupancyGridState occupancy_state = this->trainer->host.occupancy_grid_occupied_cells == train::config::nerf_grid_cells ? OccupancyGridState::Full : OccupancyGridState::Static;
        std::array<float, 3u> field_metric_extent{};
        for (std::size_t column = 0uz; column < 3uz; ++column) {
            const float x      = this->trainer->host.field_to_world_linear[column];
            const float y      = this->trainer->host.field_to_world_linear[3uz + column];
            const float z      = this->trainer->host.field_to_world_linear[6uz + column];
            const float extent = std::sqrt(x * x + y * y + z * z);
            if (!std::isfinite(extent) || extent <= 0.0f) throw std::runtime_error{"Field domain metric extent is invalid."};
            field_metric_extent[column] = extent;
        }
        return TrainingDomainView{
            .field_min           = {0.0f, 0.0f, 0.0f},
            .field_max           = {1.0f, 1.0f, 1.0f},
            .field_metric_extent = field_metric_extent,
            .coordinate_space    = TrainingCoordinateSpace::Field,
            .occupancy_state     = occupancy_state,
        };
    }

    TrainingBatchDiagnostics Inspector::training_batch_diagnostics() const {
        if (this->trainer->host.current_step == 0u || this->trainer->host.measured_sample_count == 0u) throw std::runtime_error{"training batch diagnostics require an initialized training batch."};
        if (this->trainer->device.compacted_sample_coords == nullptr) throw std::runtime_error{"training batch compacted sample buffer is null."};
        cuda::TrainingBatchDiagnostics diagnostics{};
        cuda::read_training_batch_diagnostics(this->trainer->device.compacted_sample_coords, this->trainer->host.measured_sample_count, diagnostics);
        return TrainingBatchDiagnostics{
            .sample_coord_min           = {diagnostics.coord_min[0u], diagnostics.coord_min[1u], diagnostics.coord_min[2u]},
            .sample_coord_max           = {diagnostics.coord_max[0u], diagnostics.coord_max[1u], diagnostics.coord_max[2u]},
            .time_min                   = diagnostics.time_min,
            .time_max                   = diagnostics.time_max,
            .dt_metric_min              = diagnostics.dt_metric_min,
            .dt_metric_mean             = diagnostics.dt_metric_mean,
            .dt_metric_max              = diagnostics.dt_metric_max,
            .metric_per_field_unit_min  = diagnostics.metric_per_field_unit_min,
            .metric_per_field_unit_mean = diagnostics.metric_per_field_unit_mean,
            .metric_per_field_unit_max  = diagnostics.metric_per_field_unit_max,
        };
    }

    TrainingModelDiagnostics Inspector::training_model_diagnostics() const {
        if (this->trainer->device.params_full_precision == nullptr || this->trainer->device.param_gradients == nullptr) throw std::runtime_error{"training model parameter buffers are null."};
        float global_rgb_param    = 0.0f;
        float global_rgb_gradient = 0.0f;
        cuda::read_float_value(this->trainer->device.params_full_precision, train::config::network_parameter_layout.global_rgb_offset, global_rgb_param);
        cuda::read_float_value(this->trainer->device.param_gradients, train::config::network_parameter_layout.global_rgb_offset, global_rgb_gradient);
        global_rgb_gradient /= train::config::optimizer_loss_scale;
        return TrainingModelDiagnostics{
            .global_rgb_param    = global_rgb_param,
            .global_rgb_color    = 0.6f + std::tanh(global_rgb_param) * 0.4f,
            .global_rgb_gradient = global_rgb_gradient,
        };
    }

    OccupancyGridDeviceView Inspector::occupancy_grid_device_view() const {
        return OccupancyGridDeviceView{
            .dimensions     = {train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size},
            .cell_count     = train::config::nerf_grid_cells,
            .bitfield       = this->trainer->device.occupancy,
            .bitfield_bytes = train::config::nerf_grid_cells / 8u,
            .occupied_cells = this->trainer->host.occupancy_grid_occupied_cells,
            .revision       = 1u,
            .encoding       = OccupancyGridEncoding::MortonBitfield,
            .initialized    = this->trainer->device.occupancy != nullptr,
        };
    }

    SamplerBatchDeviceView Inspector::sampler_batch_device_view() const {
        std::uint32_t ray_count = 0u;
        if (this->trainer->host.current_step != 0u && this->trainer->host.measured_sample_count != 0u) {
            if (this->trainer->device.ray_counter == nullptr) throw std::runtime_error{"sampler batch ray counter buffer is null."};
            cuda::read_counter(this->trainer->device.ray_counter, ray_count);
        }
        const bool initialized = this->trainer->host.current_step != 0u && ray_count != 0u && this->trainer->host.measured_sample_count != 0u;
        return SamplerBatchDeviceView{
            .current_step  = this->trainer->host.current_step,
            .ray_count     = ray_count,
            .sample_count  = this->trainer->host.measured_sample_count,
            .rays          = this->trainer->device.rays,
            .numsteps      = this->trainer->device.numsteps,
            .sample_coords = this->trainer->device.compacted_sample_coords,
            .revision      = this->trainer->host.current_step,
            .initialized   = initialized,
        };
    }

    std::expected<DensitySliceSampleStats, std::string> Inspector::sample_density_slice(DensitySliceSampleRequest request) const {
        try {
            if (this->trainer->host.current_step == 0u) throw std::runtime_error{"density slice cannot be sampled before the first training step."};
            if (request.dimensions != std::array{DensitySliceDimension, DensitySliceDimension, DensitySliceDimension}) throw std::runtime_error{"density slice dimensions must match HyFluid density slice dimension."};
            if (request.encoding != DensitySliceEncoding::MortonFloat32) throw std::runtime_error{"density slice sample only supports MortonFloat32 encoding."};
            if (request.time_count == 0u) throw std::runtime_error{"density slice time count must be positive."};
            if (request.time_index >= request.time_count) throw std::runtime_error{"density slice time index is outside time count."};
            if (request.output_density == nullptr) throw std::runtime_error{"density slice output pointer must not be null."};
            if (static_cast<std::uint64_t>(request.dimensions[0u]) > std::numeric_limits<std::uint64_t>::max() / request.dimensions[1u] / request.dimensions[2u]) throw std::runtime_error{"density slice dimensions overflow cell count."};
            const std::uint64_t cell_count = static_cast<std::uint64_t>(request.dimensions[0u]) * static_cast<std::uint64_t>(request.dimensions[1u]) * static_cast<std::uint64_t>(request.dimensions[2u]);
            if (cell_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) throw std::runtime_error{"density slice byte size overflows uint64."};
            const std::uint64_t byte_size = cell_count * sizeof(float);
            if (request.byte_size < byte_size) throw std::runtime_error{"density slice output buffer is too small."};
            if (this->trainer->device.params == nullptr || this->trainer->device.sample_coords == nullptr || this->trainer->device.network_input == nullptr || this->trainer->device.network_hidden == nullptr || this->trainer->device.network_output == nullptr || this->trainer->device.cublaslt_handle == nullptr || this->trainer->device.cublaslt_workspace == nullptr) throw std::runtime_error{"density slice trainer device buffers are not initialized."};
            const float time                    = request.time_count == 1u ? 0.0f : static_cast<float>(request.time_index) / static_cast<float>(request.time_count - 1u);
            float density_min                   = 0.0f;
            float density_max                   = 0.0f;
            float density_mean                  = 0.0f;
            std::uint64_t density_nonzero_count = 0u;
            cuda::sample_density_slice(request.dimensions[0u], request.dimensions[1u], request.dimensions[2u], time, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.network_input, this->trainer->device.network_hidden, this->trainer->device.network_output, this->trainer->device.cublaslt_handle, this->trainer->device.cublaslt_workspace, request.output_density, density_min, density_max, density_mean, density_nonzero_count);
            return DensitySliceSampleStats{
                .dimensions            = request.dimensions,
                .byte_size             = byte_size,
                .revision              = (static_cast<std::uint64_t>(this->trainer->host.current_step) << 32u) | static_cast<std::uint64_t>(request.time_index),
                .density_min           = density_min,
                .density_max           = density_max,
                .density_mean          = density_mean,
                .density_nonzero_count = density_nonzero_count,
                .encoding              = DensitySliceEncoding::MortonFloat32,
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<SamplerVisualizationStats, std::string> Inspector::write_sampler_visualization(SamplerVisualizationRequest request) const {
        try {
            const SamplerBatchDeviceView view = this->sampler_batch_device_view();
            if (!view.initialized) throw std::runtime_error{"sampler batch has not been initialized."};
            if (view.rays == nullptr || view.numsteps == nullptr || view.sample_coords == nullptr) throw std::runtime_error{"sampler batch device view contains null buffers."};
            if (request.time_count == 0u) throw std::runtime_error{"sampler visualization time count must be positive."};
            if (request.time_index >= request.time_count) throw std::runtime_error{"sampler visualization time index is outside time count."};
            if (!std::isfinite(request.point_radius) || request.point_radius <= 0.0f) throw std::runtime_error{"sampler point radius must be finite and positive."};
            if (!std::isfinite(request.ray_width) || request.ray_width <= 0.0f) throw std::runtime_error{"sampler ray width must be finite and positive."};
            if (request.width_mode > 1u) throw std::runtime_error{"sampler ray width mode is invalid."};
            if (view.sample_count > std::numeric_limits<std::uint64_t>::max() / SamplerPointInstanceBytes) throw std::runtime_error{"sampler point byte size overflows uint64."};
            if (view.ray_count > std::numeric_limits<std::uint64_t>::max() / SamplerSegmentInstanceBytes) throw std::runtime_error{"sampler segment byte size overflows uint64."};
            const std::uint64_t expected_point_bytes   = static_cast<std::uint64_t>(view.sample_count) * SamplerPointInstanceBytes;
            const std::uint64_t expected_segment_bytes = static_cast<std::uint64_t>(view.ray_count) * SamplerSegmentInstanceBytes;
            const bool writes_points                   = request.point_instances != nullptr || request.point_byte_size != 0u;
            const bool writes_segments                 = request.segment_instances != nullptr || request.segment_byte_size != 0u;
            if (!writes_points && !writes_segments) throw std::runtime_error{"sampler visualization request must include points or segments."};
            if (writes_points && request.point_instances == nullptr) throw std::runtime_error{"sampler point output pointer must not be null."};
            if (writes_points && request.point_byte_size < expected_point_bytes) throw std::runtime_error{"sampler point output buffer is too small."};
            if (writes_segments && request.segment_instances == nullptr) throw std::runtime_error{"sampler segment output pointer must not be null."};
            if (writes_segments && request.segment_byte_size < expected_segment_bytes) throw std::runtime_error{"sampler segment output buffer is too small."};
            std::uint32_t point_count = 0u;
            std::uint32_t ray_count   = 0u;
            cuda::fill_sampler_visualization(view.ray_count, view.sample_count, view.rays, view.numsteps, view.sample_coords, request.time_count, request.time_index, request.point_radius, request.ray_width, request.width_mode, request.point_instances, request.point_byte_size, request.segment_instances, request.segment_byte_size, point_count, ray_count);
            return SamplerVisualizationStats{
                .ray_count         = writes_segments ? ray_count : 0u,
                .point_count       = writes_points ? point_count : 0u,
                .point_byte_size   = writes_points ? static_cast<std::uint64_t>(point_count) * SamplerPointInstanceBytes : 0u,
                .segment_byte_size = writes_segments ? static_cast<std::uint64_t>(ray_count) * SamplerSegmentInstanceBytes : 0u,
                .revision          = (view.revision << 32u) | static_cast<std::uint64_t>(request.time_index),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::inspector
