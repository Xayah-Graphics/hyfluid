module;
#include "hyfluid.train.h"
#include "hyfluid.inspector.h"
module hyfluid.inspector;
import std;
import hyfluid.train;

namespace hyfluid::inspector {
    Inspector::Inspector(const train::HyFluid& trainer) : trainer{std::addressof(trainer)} {}

    SamplerBatchDeviceView Inspector::sampler_batch_device_view() const {
        std::uint32_t ray_count = 0u;
        if (this->trainer->host.current_step != 0u && this->trainer->host.measured_sample_count != 0u) {
            if (this->trainer->device.ray_counter == nullptr) throw std::runtime_error{"sampler batch ray counter buffer is null."};
            cuda::read_counter(this->trainer->device.ray_counter, ray_count);
        }
        const bool initialized = this->trainer->host.current_step != 0u && ray_count != 0u && this->trainer->host.measured_sample_count != 0u;
        return SamplerBatchDeviceView{
            .current_step = this->trainer->host.current_step,
            .ray_count = ray_count,
            .sample_count = this->trainer->host.measured_sample_count,
            .rays = this->trainer->device.rays,
            .numsteps = this->trainer->device.numsteps,
            .sample_coords = this->trainer->device.sample_coords,
            .revision = this->trainer->host.current_step,
            .initialized = initialized,
        };
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
            const std::uint64_t expected_point_bytes = static_cast<std::uint64_t>(view.sample_count) * SamplerPointInstanceBytes;
            const std::uint64_t expected_segment_bytes = static_cast<std::uint64_t>(view.ray_count) * SamplerSegmentInstanceBytes;
            const bool writes_points = request.point_instances != nullptr || request.point_byte_size != 0u;
            const bool writes_segments = request.segment_instances != nullptr || request.segment_byte_size != 0u;
            if (!writes_points && !writes_segments) throw std::runtime_error{"sampler visualization request must include points or segments."};
            if (writes_points && request.point_instances == nullptr) throw std::runtime_error{"sampler point output pointer must not be null."};
            if (writes_points && request.point_byte_size < expected_point_bytes) throw std::runtime_error{"sampler point output buffer is too small."};
            if (writes_segments && request.segment_instances == nullptr) throw std::runtime_error{"sampler segment output pointer must not be null."};
            if (writes_segments && request.segment_byte_size < expected_segment_bytes) throw std::runtime_error{"sampler segment output buffer is too small."};
            std::uint32_t point_count = 0u;
            std::uint32_t ray_count = 0u;
            cuda::fill_sampler_visualization(view.ray_count, view.sample_count, view.rays, view.numsteps, view.sample_coords, request.time_count, request.time_index, request.point_radius, request.ray_width, request.width_mode, request.point_instances, request.point_byte_size, request.segment_instances, request.segment_byte_size, point_count, ray_count);
            return SamplerVisualizationStats{
                .ray_count = writes_segments ? ray_count : 0u,
                .point_count = writes_points ? point_count : 0u,
                .point_byte_size = writes_points ? static_cast<std::uint64_t>(point_count) * SamplerPointInstanceBytes : 0u,
                .segment_byte_size = writes_segments ? static_cast<std::uint64_t>(ray_count) * SamplerSegmentInstanceBytes : 0u,
                .revision = (view.revision << 32u) | static_cast<std::uint64_t>(request.time_index),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::inspector
