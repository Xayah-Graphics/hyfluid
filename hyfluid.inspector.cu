#include "hyfluid.train.config.h"
#include "hyfluid.inspector.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hyfluid::cuda {
    namespace {
        struct SamplerPointInstance final {
            float position_radius[4]{};
            float color[4]{};
        };

        struct SamplerSegmentInstance final {
            float start_width[4]{};
            float end[3]{};
            std::uint32_t flags{};
            float color[4]{};
        };

        static_assert(sizeof(SamplerPointInstance) == 8u * sizeof(float));
        static_assert(sizeof(SamplerSegmentInstance) == 12u * sizeof(float));

        __device__ std::uint32_t sample_time_index(const float time, const std::uint32_t time_count) {
            if (time_count == 1u) return 0u;
            const float normalized_time = fminf(1.0f, fmaxf(0.0f, time));
            const float rounded_time    = floorf(normalized_time * static_cast<float>(time_count - 1u) + 0.5f);
            if (rounded_time <= 0.0f) return 0u;
            if (rounded_time >= static_cast<float>(time_count - 1u)) return time_count - 1u;
            return static_cast<std::uint32_t>(rounded_time);
        }

        __global__ void fill_sampler_points_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, const std::uint32_t time_count, const std::uint32_t time_index, const float point_radius, SamplerPointInstance* __restrict__ point_instances, std::uint32_t* __restrict__ point_counter) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const float* coord = sample_coords + static_cast<std::uint64_t>(i) * train::config::sample_coord_floats;
            if (sample_time_index(coord[3u], time_count) != time_index) return;

            const std::uint32_t output_index = atomicAdd(point_counter, 1u);
            const float time = fminf(1.0f, fmaxf(0.0f, coord[3u]));
            SamplerPointInstance point{};
            point.position_radius[0u] = coord[2u];
            point.position_radius[1u] = coord[1u];
            point.position_radius[2u] = coord[0u];
            point.position_radius[3u] = point_radius;
            point.color[0u] = 0.10f + 0.90f * time;
            point.color[1u] = 0.76f - 0.34f * time;
            point.color[2u] = 1.00f - 0.82f * time;
            point.color[3u] = 0.95f;
            point_instances[output_index] = point;
        }

        __global__ void fill_sampler_segments_kernel(const std::uint32_t ray_count, const std::uint32_t sample_count, const float* __restrict__ rays, const std::uint32_t* __restrict__ numsteps, const float* __restrict__ sample_coords, const std::uint32_t time_count, const std::uint32_t time_index, const float ray_width, const std::uint32_t width_mode, SamplerSegmentInstance* __restrict__ segment_instances, std::uint32_t* __restrict__ ray_counter) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= ray_count) return;

            const std::uint32_t steps = numsteps[i * 2u + 0u];
            const std::uint32_t base = numsteps[i * 2u + 1u];
            if (steps == 0u || base >= sample_count || steps > sample_count - base) return;

            const float* start = sample_coords + static_cast<std::uint64_t>(base) * train::config::sample_coord_floats;
            if (sample_time_index(start[3u], time_count) != time_index) return;

            const std::uint32_t output_index = atomicAdd(ray_counter, 1u);
            const float* end = sample_coords + static_cast<std::uint64_t>(base + steps - 1u) * train::config::sample_coord_floats;
            SamplerSegmentInstance segment{};
            segment.start_width[3u] = ray_width;
            segment.flags = width_mode;
            segment.start_width[0u] = start[2u];
            segment.start_width[1u] = start[1u];
            segment.start_width[2u] = start[0u];
            segment.end[0u] = end[2u];
            segment.end[1u] = end[1u];
            segment.end[2u] = end[0u];

            if (steps == 1u) {
                const float* ray = rays + static_cast<std::uint64_t>(i) * train::config::ray_floats;
                segment.end[0u] = segment.start_width[0u] + ray[5u] * train::config::min_cone_stepsize;
                segment.end[1u] = segment.start_width[1u] + ray[4u] * train::config::min_cone_stepsize;
                segment.end[2u] = segment.start_width[2u] + ray[3u] * train::config::min_cone_stepsize;
            }

            const float time = fminf(1.0f, fmaxf(0.0f, start[3u]));
            const float heat = fminf(1.0f, static_cast<float>(steps) / 128.0f);
            segment.color[0u] = 0.18f + 0.82f * time;
            segment.color[1u] = 0.42f + 0.40f * heat;
            segment.color[2u] = 0.95f - 0.72f * time;
            segment.color[3u] = 0.52f;
            segment_instances[output_index] = segment;
        }
    } // namespace

    void fill_sampler_visualization(const std::uint32_t ray_count, const std::uint32_t sample_count, const float* const rays, const std::uint32_t* const numsteps, const float* const sample_coords, const std::uint32_t time_count, const std::uint32_t time_index, const float point_radius, const float ray_width, const std::uint32_t width_mode, std::byte* const point_instances, const std::uint64_t point_byte_size, std::byte* const segment_instances, const std::uint64_t segment_byte_size, std::uint32_t& out_point_count, std::uint32_t& out_ray_count) {
        if (ray_count == 0u || sample_count == 0u || rays == nullptr || numsteps == nullptr || sample_coords == nullptr) throw std::runtime_error{"invalid sampler visualization input."};
        if (time_count == 0u || time_index >= time_count) throw std::runtime_error{"invalid sampler visualization time filter."};
        if (width_mode > 1u) throw std::runtime_error{"invalid sampler visualization width mode."};
        out_point_count = 0u;
        out_ray_count = 0u;

        std::uint32_t* counters = nullptr;
        try {
            if (const cudaError_t status = cudaMalloc(&counters, 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler visualization counters failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemset(counters, 0, 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler visualization counters failed: "} + cudaGetErrorString(status)};

            if (point_instances != nullptr) {
                const std::uint64_t expected_point_bytes = static_cast<std::uint64_t>(sample_count) * sizeof(SamplerPointInstance);
                if (point_byte_size < expected_point_bytes) throw std::runtime_error{"sampler point output byte size is too small."};
                fill_sampler_points_kernel<<<(sample_count + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(sample_count, sample_coords, time_count, time_index, point_radius, reinterpret_cast<SamplerPointInstance*>(point_instances), counters);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_sampler_points_kernel failed: "} + cudaGetErrorString(status)};
            }
            if (segment_instances != nullptr) {
                const std::uint64_t expected_segment_bytes = static_cast<std::uint64_t>(ray_count) * sizeof(SamplerSegmentInstance);
                if (segment_byte_size < expected_segment_bytes) throw std::runtime_error{"sampler segment output byte size is too small."};
                fill_sampler_segments_kernel<<<(ray_count + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(ray_count, sample_count, rays, numsteps, sample_coords, time_count, time_index, ray_width, width_mode, reinterpret_cast<SamplerSegmentInstance*>(segment_instances), counters + 1u);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_sampler_segments_kernel failed: "} + cudaGetErrorString(status)};
            }
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after sampler visualization failed: "} + cudaGetErrorString(status)};

            std::uint32_t counts[2]{};
            if (const cudaError_t status = cudaMemcpy(counts, counters, 2u * sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy sampler visualization counters failed: "} + cudaGetErrorString(status)};
            out_point_count = counts[0u];
            out_ray_count = counts[1u];
            const cudaError_t free_status = cudaFree(counters);
            counters = nullptr;
            if (free_status != cudaSuccess) throw std::runtime_error{std::string{"cudaFree sampler visualization counters failed: "} + cudaGetErrorString(free_status)};
        } catch (...) {
            if (counters != nullptr) cudaFree(counters);
            throw;
        }
    }
} // namespace hyfluid::cuda
