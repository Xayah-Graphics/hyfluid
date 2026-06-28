#include "hyfluid.train.h"
#include "hyfluid.train.config.h"
#include <cuda/std/algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hyfluid::cuda {
    namespace {
        struct Pcg32 final {
            std::uint64_t state = train::config::pcg32_default_state;
            std::uint64_t inc   = train::config::pcg32_default_stream;

            Pcg32() = default;

            __host__ __device__ explicit Pcg32(const std::uint64_t initstate, const std::uint64_t initseq = 1u) {
                this->state = 0u;
                this->inc   = (initseq << 1u) | 1u;
                this->next_uint();
                this->state += initstate;
                this->next_uint();
            }

            __host__ __device__ std::uint32_t next_uint() {
                const std::uint64_t oldstate = this->state;
                this->state                  = oldstate * train::config::pcg32_mult + this->inc;
                const auto xorshifted        = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
                const auto rot               = static_cast<std::uint32_t>(oldstate >> 59u);
                return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
            }

            __host__ __device__ float next_float() {
                return static_cast<float>(this->next_uint() >> 8u) * (1.0f / 16777216.0f);
            }

            __host__ __device__ void advance(std::uint64_t delta) {
                std::uint64_t cur_mult = train::config::pcg32_mult;
                std::uint64_t cur_plus = this->inc;
                std::uint64_t acc_mult = 1u;
                std::uint64_t acc_plus = 0u;
                while (delta > 0u) {
                    if ((delta & 1u) != 0u) {
                        acc_mult *= cur_mult;
                        acc_plus = acc_plus * cur_mult + cur_plus;
                    }
                    cur_plus = (cur_mult + 1u) * cur_plus;
                    cur_mult *= cur_mult;
                    delta >>= 1u;
                }
                this->state = acc_mult * this->state + acc_plus;
            }
        };

        __device__ bool unit_aabb_contains(const float3 pos) {
            return pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f && pos.z >= 0.0f && pos.z <= 1.0f;
        }

        __device__ bool intersect_unit_aabb(const float3 origin, const float3 direction, float& out_tmin) {
            const float3 inv_dir = {1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z};
            const float3 t0      = {-origin.x * inv_dir.x, -origin.y * inv_dir.y, -origin.z * inv_dir.z};
            const float3 t1      = {(1.0f - origin.x) * inv_dir.x, (1.0f - origin.y) * inv_dir.y, (1.0f - origin.z) * inv_dir.z};

            const float tx_min = fminf(t0.x, t1.x);
            const float tx_max = fmaxf(t0.x, t1.x);
            const float ty_min = fminf(t0.y, t1.y);
            const float ty_max = fmaxf(t0.y, t1.y);
            const float tz_min = fminf(t0.z, t1.z);
            const float tz_max = fmaxf(t0.z, t1.z);

            const float tmin = fmaxf(fmaxf(tx_min, ty_min), tz_min);
            const float tmax = fminf(fminf(tx_max, ty_max), tz_max);
            out_tmin         = fmaxf(tmin, 0.0f);
            return tmax >= out_tmin;
        }

        __device__ bool is_occupancy_grid_cell_occupied(const float3 pos, const std::uint8_t* occupancy) {
            const int x = static_cast<int>(pos.x * static_cast<float>(train::config::nerf_grid_size));
            const int y = static_cast<int>(pos.y * static_cast<float>(train::config::nerf_grid_size));
            const int z = static_cast<int>(pos.z * static_cast<float>(train::config::nerf_grid_size));
            if (x < 0 || x >= static_cast<int>(train::config::nerf_grid_size) || y < 0 || y >= static_cast<int>(train::config::nerf_grid_size) || z < 0 || z >= static_cast<int>(train::config::nerf_grid_size)) return false;
            auto morton_x             = static_cast<std::uint32_t>(x);
            auto morton_y             = static_cast<std::uint32_t>(y);
            auto morton_z             = static_cast<std::uint32_t>(z);
            morton_x                  = (morton_x * 0x00010001u) & 0xFF0000FFu;
            morton_x                  = (morton_x * 0x00000101u) & 0x0F00F00Fu;
            morton_x                  = (morton_x * 0x00000011u) & 0xC30C30C3u;
            morton_x                  = (morton_x * 0x00000005u) & 0x49249249u;
            morton_y                  = (morton_y * 0x00010001u) & 0xFF0000FFu;
            morton_y                  = (morton_y * 0x00000101u) & 0x0F00F00Fu;
            morton_y                  = (morton_y * 0x00000011u) & 0xC30C30C3u;
            morton_y                  = (morton_y * 0x00000005u) & 0x49249249u;
            morton_z                  = (morton_z * 0x00010001u) & 0xFF0000FFu;
            morton_z                  = (morton_z * 0x00000101u) & 0x0F00F00Fu;
            morton_z                  = (morton_z * 0x00000011u) & 0xC30C30C3u;
            morton_z                  = (morton_z * 0x00000005u) & 0x49249249u;
            const std::uint32_t index = morton_x | (morton_y << 1u) | (morton_z << 2u);
            return (occupancy[index / 8u] & (1u << (index % 8u))) != 0u;
        }

        __device__ float advance_to_next_density_voxel(const float t, const float3 pos, const float3 direction, const float3 inv_direction) {
            constexpr auto scale = static_cast<float>(train::config::nerf_grid_size);
            const float3 p       = {(pos.x - 0.5f) * scale, (pos.y - 0.5f) * scale, (pos.z - 0.5f) * scale};
            const float tx       = (floorf(p.x + 0.5f + 0.5f * copysignf(1.0f, direction.x)) - p.x) * inv_direction.x;
            const float ty       = (floorf(p.y + 0.5f + 0.5f * copysignf(1.0f, direction.y)) - p.y) * inv_direction.y;
            const float tz       = (floorf(p.z + 0.5f + 0.5f * copysignf(1.0f, direction.z)) - p.z) * inv_direction.z;
            const float t_target = t + fmaxf(fminf(fminf(tx, ty), tz) / scale, 0.0f);
            return t + ceilf(fmaxf((t_target - t) / train::config::min_cone_stepsize, 0.5f)) * train::config::min_cone_stepsize;
        }

        __global__ void generate_training_samples_kernel(const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint32_t current_step, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float* __restrict__ camera, const float* __restrict__ intrinsics, const float* __restrict__ times, const std::uint32_t* __restrict__ frame_indices, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ ray_indices_out, float* __restrict__ rays_out, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= rays_per_batch) return;

            Pcg32 rng{train::config::train_seed};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(i) * train::config::max_random_samples_per_ray);

            const std::uint32_t frame_grid_index = rng.next_uint() % frame_count;
            const std::uint32_t frame_index      = frame_indices[frame_grid_index];
            const float* frame_camera            = camera + static_cast<std::uint64_t>(frame_index) * 12u;
            const float* frame_intrinsics        = intrinsics + static_cast<std::uint64_t>(frame_index) * 4u;
            const float frame_time               = times[frame_index];

            float u = rng.next_float();
            float v = rng.next_float();
            if (train::config::snap_to_pixel_centers) {
                const std::uint32_t pixel_x = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
                const std::uint32_t pixel_y = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
                u                           = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
                v                           = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
            }
            const float ray_x       = (u * static_cast<float>(width) - frame_intrinsics[2u]) / frame_intrinsics[0u];
            const float ray_y       = (v * static_cast<float>(height) - frame_intrinsics[3u]) / frame_intrinsics[1u];
            const float3 camera_x   = {frame_camera[0], frame_camera[1], frame_camera[2]};
            const float3 camera_y   = {frame_camera[3], frame_camera[4], frame_camera[5]};
            const float3 camera_z   = {frame_camera[6], frame_camera[7], frame_camera[8]};
            const float3 ray_origin = {frame_camera[9], frame_camera[10], frame_camera[11]};
            float3 ray_direction    = {
                camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
            };
            if (ray_direction.x == 0.0f && ray_direction.y == 0.0f && ray_direction.z == 0.0f) ray_direction = camera_z;

            const float direction_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
            if (direction_length == 0.0f) return;
            const float3 ray_direction_normalized = {ray_direction.x / direction_length, ray_direction.y / direction_length, ray_direction.z / direction_length};

            float tmin = 0.0f;
            if (!intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin)) return;

            constexpr float dt         = train::config::min_cone_stepsize;
            const float start_t        = tmin + rng.next_float() * dt;
            const float3 inv_direction = {1.0f / ray_direction_normalized.x, 1.0f / ray_direction_normalized.y, 1.0f / ray_direction_normalized.z};

            std::uint32_t numsteps = 0u;
            float t                = start_t;
            float3 pos             = {};

            while (numsteps < train::config::nerf_steps) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_occupancy_grid_cell_occupied(pos, occupancy)) {
                    ++numsteps;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base + numsteps > sample_limit) return;

            const std::uint32_t ray_index = atomicAdd(ray_counter, 1u);
            ray_indices_out[ray_index]    = i;

            float* ray_out = rays_out + static_cast<std::uint64_t>(ray_index) * train::config::ray_floats;
            ray_out[0]     = ray_origin.x;
            ray_out[1]     = ray_origin.y;
            ray_out[2]     = ray_origin.z;
            ray_out[3]     = ray_direction_normalized.x;
            ray_out[4]     = ray_direction_normalized.y;
            ray_out[5]     = ray_direction_normalized.z;

            numsteps_out[ray_index * 2u + 0u] = numsteps;
            numsteps_out[ray_index * 2u + 1u] = base;

            t = start_t;
            std::uint32_t j = 0u;
            while (j < numsteps) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_occupancy_grid_cell_occupied(pos, occupancy)) {
                    float* coord = coords_out + static_cast<std::uint64_t>(base + j) * train::config::sample_coord_floats;
                    coord[0]     = pos.x;
                    coord[1]     = pos.y;
                    coord[2]     = pos.z;
                    coord[3]     = frame_time;
                    coord[4]     = dt;

                    ++j;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }
        }
    } // namespace

    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] != nullptr) cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    void upload_dataset(const std::uint8_t* const pixels, const std::size_t pixel_bytes, const float* const camera, const std::size_t camera_count, const float* const intrinsics, const std::size_t intrinsics_count, const float* const times, const std::size_t time_count, const std::uint32_t* const view_indices, const std::size_t view_index_count, const std::uint32_t* const time_indices, const std::size_t time_index_count, const std::uint32_t* const frame_indices, const std::size_t frame_index_count, const std::uint8_t*& out_pixels, const float*& out_camera, const float*& out_intrinsics, const float*& out_times, const std::uint32_t*& out_view_indices, const std::uint32_t*& out_time_indices, const std::uint32_t*& out_frame_indices) {
        out_pixels        = nullptr;
        out_camera        = nullptr;
        out_intrinsics    = nullptr;
        out_times         = nullptr;
        out_view_indices  = nullptr;
        out_time_indices  = nullptr;
        out_frame_indices = nullptr;

        if (pixels == nullptr || pixel_bytes == 0u || camera == nullptr || camera_count == 0u || intrinsics == nullptr || intrinsics_count == 0u || times == nullptr || time_count == 0u || view_indices == nullptr || view_index_count == 0u || time_indices == nullptr || time_index_count == 0u || frame_indices == nullptr || frame_index_count == 0u) throw std::runtime_error{"invalid dynamic dataset upload input."};
        if (camera_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"camera upload is too large."};
        if (intrinsics_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"intrinsics upload is too large."};
        if (time_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"time upload is too large."};
        if (view_index_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) throw std::runtime_error{"view index upload is too large."};
        if (time_index_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) throw std::runtime_error{"time index upload is too large."};
        if (frame_index_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) throw std::runtime_error{"frame index upload is too large."};

        const std::size_t camera_bytes      = camera_count * sizeof(float);
        const std::size_t intrinsics_bytes  = intrinsics_count * sizeof(float);
        const std::size_t time_bytes        = time_count * sizeof(float);
        const std::size_t view_index_bytes  = view_index_count * sizeof(std::uint32_t);
        const std::size_t time_index_bytes  = time_index_count * sizeof(std::uint32_t);
        const std::size_t frame_index_bytes = frame_index_count * sizeof(std::uint32_t);
        if (pixel_bytes > std::numeric_limits<std::size_t>::max() - camera_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        std::size_t required_bytes = pixel_bytes + camera_bytes;
        if (required_bytes > std::numeric_limits<std::size_t>::max() - intrinsics_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        required_bytes += intrinsics_bytes;
        if (required_bytes > std::numeric_limits<std::size_t>::max() - time_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        required_bytes += time_bytes;
        if (required_bytes > std::numeric_limits<std::size_t>::max() - view_index_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        required_bytes += view_index_bytes;
        if (required_bytes > std::numeric_limits<std::size_t>::max() - time_index_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        required_bytes += time_index_bytes;
        if (required_bytes > std::numeric_limits<std::size_t>::max() - frame_index_bytes) throw std::runtime_error{"dynamic dataset upload is too large."};
        required_bytes += frame_index_bytes;

        std::size_t free_bytes  = 0u;
        std::size_t total_bytes = 0u;
        if (const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemGetInfo failed: "} + cudaGetErrorString(status)};
        if (required_bytes > free_bytes) throw std::runtime_error{"dynamic dataset does not fit in available GPU memory."};

        void* uploaded_pixels = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_pixels, pixel_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pixels failed: "} + cudaGetErrorString(status)};
        out_pixels = static_cast<std::uint8_t*>(uploaded_pixels);
        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint8_t*>(out_pixels), pixels, pixel_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy pixels failed: "} + cudaGetErrorString(status)};

        void* uploaded_camera = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_camera, camera_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc camera failed: "} + cudaGetErrorString(status)};
        out_camera = static_cast<float*>(uploaded_camera);
        if (const cudaError_t status = cudaMemcpy(const_cast<float*>(out_camera), camera, camera_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy camera failed: "} + cudaGetErrorString(status)};

        void* uploaded_intrinsics = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_intrinsics, intrinsics_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc intrinsics failed: "} + cudaGetErrorString(status)};
        out_intrinsics = static_cast<float*>(uploaded_intrinsics);
        if (const cudaError_t status = cudaMemcpy(const_cast<float*>(out_intrinsics), intrinsics, intrinsics_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy intrinsics failed: "} + cudaGetErrorString(status)};

        void* uploaded_times = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_times, time_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc times failed: "} + cudaGetErrorString(status)};
        out_times = static_cast<float*>(uploaded_times);
        if (const cudaError_t status = cudaMemcpy(const_cast<float*>(out_times), times, time_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy times failed: "} + cudaGetErrorString(status)};

        void* uploaded_view_indices = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_view_indices, view_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc view indices failed: "} + cudaGetErrorString(status)};
        out_view_indices = static_cast<std::uint32_t*>(uploaded_view_indices);
        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint32_t*>(out_view_indices), view_indices, view_index_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy view indices failed: "} + cudaGetErrorString(status)};

        void* uploaded_time_indices = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_time_indices, time_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc time indices failed: "} + cudaGetErrorString(status)};
        out_time_indices = static_cast<std::uint32_t*>(uploaded_time_indices);
        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint32_t*>(out_time_indices), time_indices, time_index_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy time indices failed: "} + cudaGetErrorString(status)};

        void* uploaded_frame_indices = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_frame_indices, frame_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc frame indices failed: "} + cudaGetErrorString(status)};
        out_frame_indices = static_cast<std::uint32_t*>(uploaded_frame_indices);
        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint32_t*>(out_frame_indices), frame_indices, frame_index_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy frame indices failed: "} + cudaGetErrorString(status)};
    }

    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy, std::uint32_t*& out_occupancy_grid_occupied_count) {
        out_sample_coords = nullptr;
        out_rays = nullptr;
        out_ray_indices = nullptr;
        out_numsteps = nullptr;
        out_ray_counter = nullptr;
        out_sample_counter = nullptr;
        out_occupancy = nullptr;
        out_occupancy_grid_occupied_count = nullptr;

        constexpr std::size_t sample_coord_bytes = static_cast<std::size_t>(train::config::max_samples) * train::config::sample_coord_floats * sizeof(float);
        constexpr std::size_t ray_bytes = static_cast<std::size_t>(train::config::initial_rays_per_batch) * train::config::ray_floats * sizeof(float);
        constexpr std::size_t ray_index_bytes = static_cast<std::size_t>(train::config::initial_rays_per_batch) * sizeof(std::uint32_t);
        constexpr std::size_t numstep_bytes = static_cast<std::size_t>(train::config::initial_rays_per_batch) * 2u * sizeof(std::uint32_t);
        constexpr std::size_t occupancy_bytes = train::config::nerf_grid_cells / 8u;

        std::size_t required_bytes = sample_coord_bytes + ray_bytes + ray_index_bytes + numstep_bytes + occupancy_bytes + 3u * sizeof(std::uint32_t);
        std::size_t free_bytes  = 0u;
        std::size_t total_bytes = 0u;
        if (const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemGetInfo failed: "} + cudaGetErrorString(status)};
        if (required_bytes > free_bytes) throw std::runtime_error{"sampler buffers do not fit in available GPU memory."};

        if (const cudaError_t status = cudaMalloc(&out_sample_coords, sample_coord_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample coords failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rays, ray_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler rays failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_indices, ray_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray indices failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_numsteps, numstep_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler numsteps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_occupancy, occupancy_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_occupancy_grid_occupied_count, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy grid occupied count failed: "} + cudaGetErrorString(status)};

        if (const cudaError_t status = cudaMemset(out_ray_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler sample counter failed: "} + cudaGetErrorString(status)};
    }

    void set_occupancy_grid_full(std::uint8_t* const occupancy, std::uint32_t* const occupancy_grid_occupied_count) {
        if (occupancy == nullptr || occupancy_grid_occupied_count == nullptr) throw std::runtime_error{"invalid occupancy grid full update input."};
        constexpr std::size_t occupancy_bytes = train::config::nerf_grid_cells / 8u;
        if (const cudaError_t status = cudaMemset(occupancy, 0xFF, occupancy_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset full occupancy failed: "} + cudaGetErrorString(status)};
        constexpr std::uint32_t occupied_count = train::config::nerf_grid_cells;
        if (const cudaError_t status = cudaMemcpy(occupancy_grid_occupied_count, &occupied_count, sizeof(std::uint32_t), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy full occupancy count failed: "} + cudaGetErrorString(status)};
    }

    void read_counter(const std::uint32_t* const counter, std::uint32_t& value) {
        if (counter == nullptr) throw std::runtime_error{"counter read input is null."};
        if (const cudaError_t status = cudaMemcpy(&value, counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy counter read failed: "} + cudaGetErrorString(status)};
    }

    void sample_training_batch(const float* const camera, const float* const intrinsics, const float* const times, const std::uint32_t* const frame_indices, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const std::uint32_t current_step, const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint8_t* const occupancy, float* const sample_coords, float* const rays, std::uint32_t* const ray_indices, std::uint32_t* const numsteps, std::uint32_t* const ray_counter, std::uint32_t* const sample_counter) {
        if (camera == nullptr || intrinsics == nullptr || times == nullptr || frame_indices == nullptr || occupancy == nullptr || sample_coords == nullptr || rays == nullptr || ray_indices == nullptr || numsteps == nullptr || ray_counter == nullptr || sample_counter == nullptr) throw std::runtime_error{"invalid training sampler input."};
        if (frame_count == 0u || width == 0u || height == 0u || rays_per_batch == 0u || rays_per_batch > train::config::initial_rays_per_batch || sample_limit == 0u || sample_limit > train::config::max_samples) throw std::runtime_error{"invalid training sampler shape."};
        if (const cudaError_t status = cudaMemset(ray_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler sample counter failed: "} + cudaGetErrorString(status)};

        const std::uint32_t blocks = (rays_per_batch + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        generate_training_samples_kernel<<<blocks, train::config::threads_per_block>>>(rays_per_batch, sample_limit, current_step, frame_count, width, height, camera, intrinsics, times, frame_indices, occupancy, ray_counter, sample_counter, ray_indices, rays, numsteps, sample_coords);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate_training_samples_kernel failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after generate_training_samples_kernel failed: "} + cudaGetErrorString(status)};
    }
} // namespace hyfluid::cuda
