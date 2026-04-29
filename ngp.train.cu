#include "ngp.train.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>

namespace hyfluid::cuda {
    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] != nullptr) cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    namespace {
        inline constexpr std::uint32_t THREADS_PER_BLOCK = 128u;
        inline constexpr std::uint32_t CAMERA_FLOATS     = 17u;
        inline constexpr float BBOX_MIN_X                = 0.15f;
        inline constexpr float BBOX_MIN_Y                = 0.0f;
        inline constexpr float BBOX_MIN_Z                = 0.15f;
        inline constexpr float BBOX_MAX_X                = 0.85f;
        inline constexpr float BBOX_MAX_Y                = 1.0f;
        inline constexpr float BBOX_MAX_Z                = 0.85f;
        inline constexpr float OPTIMIZER_BETA1           = 0.9f;
        inline constexpr float OPTIMIZER_BETA2           = 0.99f;
        inline constexpr float OPTIMIZER_EPSILON         = 1e-15f;
        inline constexpr float OPTIMIZER_WEIGHT_DECAY    = 1e-6f;
        inline constexpr float OCCUPANCY_EMA_DECAY       = 0.95f;
        inline constexpr float OCCUPANCY_THRESHOLD       = 1e-8f;

        __device__ float clamp01(const float value) {
            return fminf(fmaxf(value, 0.0f), 1.0f);
        }

        __device__ std::uint32_t hash4(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t t) {
            return x ^ (y * 2654435761u) ^ (z * 805459861u) ^ (t * 3674653429u);
        }

        __device__ std::uint32_t grid_index4(const std::uint32_t dense, const std::uint32_t entries, const std::uint32_t rx, const std::uint32_t ry, const std::uint32_t rz, const std::uint32_t rt, const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t t) {
            if (dense != 0u) {
                std::uint64_t index = x;
                index += static_cast<std::uint64_t>(rx + 1u) * y;
                index += static_cast<std::uint64_t>(rx + 1u) * static_cast<std::uint64_t>(ry + 1u) * z;
                index += static_cast<std::uint64_t>(rx + 1u) * static_cast<std::uint64_t>(ry + 1u) * static_cast<std::uint64_t>(rz + 1u) * t;
                return static_cast<std::uint32_t>(index % entries);
            }
            return hash4(x, y, z, t) % entries;
        }

        __device__ float sample_video_channel(const std::uint8_t* const pixels, const std::uint32_t view, const std::uint32_t frame, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const std::uint32_t x, const std::uint32_t y, const std::uint32_t channel) {
            const std::uint64_t pixel = (static_cast<std::uint64_t>(view) * frame_count + frame) * static_cast<std::uint64_t>(width) * height + static_cast<std::uint64_t>(y) * width + x;
            return static_cast<float>(pixels[pixel * 3ull + channel]) * (1.0f / 255.0f);
        }

        __device__ void sample_video_rgb(const std::uint8_t* const pixels, const std::uint32_t view, const float frame_value, const float pixel_x, const float pixel_y, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, float& out_r, float& out_g, float& out_b) {
            const float clamped_frame  = fminf(fmaxf(frame_value, 0.0f), static_cast<float>(frame_count - 1u));
            const std::uint32_t frame0 = static_cast<std::uint32_t>(floorf(clamped_frame));
            const std::uint32_t frame1 = min(frame0 + 1u, frame_count - 1u);
            const float frame_lerp     = clamped_frame - static_cast<float>(frame0);
            const float x              = fminf(fmaxf(pixel_x, 0.0f), static_cast<float>(width - 1u));
            const float y              = fminf(fmaxf(pixel_y, 0.0f), static_cast<float>(height - 1u));
            const std::uint32_t x0     = static_cast<std::uint32_t>(floorf(x));
            const std::uint32_t y0     = static_cast<std::uint32_t>(floorf(y));
            const std::uint32_t x1     = min(x0 + 1u, width - 1u);
            const std::uint32_t y1     = min(y0 + 1u, height - 1u);
            const float tx             = x - static_cast<float>(x0);
            const float ty             = y - static_cast<float>(y0);

            float channels[3] = {};
            for (std::uint32_t channel = 0u; channel < 3u; ++channel) {
                const float f00_0     = sample_video_channel(pixels, view, frame0, frame_count, width, height, x0, y0, channel);
                const float f10_0     = sample_video_channel(pixels, view, frame0, frame_count, width, height, x1, y0, channel);
                const float f01_0     = sample_video_channel(pixels, view, frame0, frame_count, width, height, x0, y1, channel);
                const float f11_0     = sample_video_channel(pixels, view, frame0, frame_count, width, height, x1, y1, channel);
                const float spatial_0 = (1.0f - ty) * ((1.0f - tx) * f00_0 + tx * f10_0) + ty * ((1.0f - tx) * f01_0 + tx * f11_0);

                const float f00_1     = sample_video_channel(pixels, view, frame1, frame_count, width, height, x0, y0, channel);
                const float f10_1     = sample_video_channel(pixels, view, frame1, frame_count, width, height, x1, y0, channel);
                const float f01_1     = sample_video_channel(pixels, view, frame1, frame_count, width, height, x0, y1, channel);
                const float f11_1     = sample_video_channel(pixels, view, frame1, frame_count, width, height, x1, y1, channel);
                const float spatial_1 = (1.0f - ty) * ((1.0f - tx) * f00_1 + tx * f10_1) + ty * ((1.0f - tx) * f01_1 + tx * f11_1);
                channels[channel]     = (1.0f - frame_lerp) * spatial_0 + frame_lerp * spatial_1;
            }
            out_r = channels[0];
            out_g = channels[1];
            out_b = channels[2];
        }

        __device__ void encode_hash4(const float x, const float y, const float z, const float t, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const float* const params, float* const encoded) {
            for (std::uint32_t level = 0u; level < HASH_LEVELS; ++level) {
                const std::uint32_t rx = hash_resolutions[level * 4u + 0u];
                const std::uint32_t ry = hash_resolutions[level * 4u + 1u];
                const std::uint32_t rz = hash_resolutions[level * 4u + 2u];
                const std::uint32_t rt = hash_resolutions[level * 4u + 3u];
                const float scaled_x   = x * static_cast<float>(rx);
                const float scaled_y   = y * static_cast<float>(ry);
                const float scaled_z   = z * static_cast<float>(rz);
                const float scaled_t   = t * static_cast<float>(rt);
                std::uint32_t grid_x   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_x)), 0), static_cast<int>(rx - 1u)));
                std::uint32_t grid_y   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_y)), 0), static_cast<int>(ry - 1u)));
                std::uint32_t grid_z   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_z)), 0), static_cast<int>(rz - 1u)));
                std::uint32_t grid_t   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_t)), 0), static_cast<int>(rt - 1u)));
                const float frac_x     = clamp01(scaled_x - static_cast<float>(grid_x));
                const float frac_y     = clamp01(scaled_y - static_cast<float>(grid_y));
                const float frac_z     = clamp01(scaled_z - static_cast<float>(grid_z));
                const float frac_t     = clamp01(scaled_t - static_cast<float>(grid_t));
                float result0          = 0.0f;
                float result1          = 0.0f;
                for (std::uint32_t corner = 0u; corner < 16u; ++corner) {
                    const bool high_x         = (corner & 1u) != 0u;
                    const bool high_y         = (corner & 2u) != 0u;
                    const bool high_z         = (corner & 4u) != 0u;
                    const bool high_t         = (corner & 8u) != 0u;
                    const float weight        = (high_x ? frac_x : 1.0f - frac_x) * (high_y ? frac_y : 1.0f - frac_y) * (high_z ? frac_z : 1.0f - frac_z) * (high_t ? frac_t : 1.0f - frac_t);
                    const std::uint32_t index = hash_offsets[level] + grid_index4(hash_dense[level], hash_entries[level], rx, ry, rz, rt, grid_x + static_cast<std::uint32_t>(high_x), grid_y + static_cast<std::uint32_t>(high_y), grid_z + static_cast<std::uint32_t>(high_z), grid_t + static_cast<std::uint32_t>(high_t)) * HASH_FEATURES_PER_LEVEL;
                    result0 += weight * params[index + 0u];
                    result1 += weight * params[index + 1u];
                }
                encoded[level * HASH_FEATURES_PER_LEVEL + 0u] = result0;
                encoded[level * HASH_FEATURES_PER_LEVEL + 1u] = result1;
            }
        }

        __device__ float forward_density(const float* const encoded, const float* const params, const std::uint32_t w1_offset, const std::uint32_t w2_offset, float* const hidden_pre, float* const hidden) {
            for (std::uint32_t row = 0u; row < MLP_HIDDEN_WIDTH; ++row) {
                float sum = 0.0f;
                for (std::uint32_t column = 0u; column < HASH_INPUT_WIDTH; ++column) sum += params[w1_offset + row * HASH_INPUT_WIDTH + column] * encoded[column];
                hidden_pre[row] = sum;
                hidden[row]     = fmaxf(sum, 0.0f);
            }
            float output = 0.0f;
            for (std::uint32_t row = 0u; row < MLP_HIDDEN_WIDTH; ++row) output += params[w2_offset + row] * hidden[row];
            return fmaxf(output, 0.0f);
        }

        __device__ void backward_hash4(const float x, const float y, const float z, const float t, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const float* const encoded_grad, float* const gradients) {
            for (std::uint32_t level = 0u; level < HASH_LEVELS; ++level) {
                const std::uint32_t rx = hash_resolutions[level * 4u + 0u];
                const std::uint32_t ry = hash_resolutions[level * 4u + 1u];
                const std::uint32_t rz = hash_resolutions[level * 4u + 2u];
                const std::uint32_t rt = hash_resolutions[level * 4u + 3u];
                const float scaled_x   = x * static_cast<float>(rx);
                const float scaled_y   = y * static_cast<float>(ry);
                const float scaled_z   = z * static_cast<float>(rz);
                const float scaled_t   = t * static_cast<float>(rt);
                std::uint32_t grid_x   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_x)), 0), static_cast<int>(rx - 1u)));
                std::uint32_t grid_y   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_y)), 0), static_cast<int>(ry - 1u)));
                std::uint32_t grid_z   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_z)), 0), static_cast<int>(rz - 1u)));
                std::uint32_t grid_t   = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(scaled_t)), 0), static_cast<int>(rt - 1u)));
                const float frac_x     = clamp01(scaled_x - static_cast<float>(grid_x));
                const float frac_y     = clamp01(scaled_y - static_cast<float>(grid_y));
                const float frac_z     = clamp01(scaled_z - static_cast<float>(grid_z));
                const float frac_t     = clamp01(scaled_t - static_cast<float>(grid_t));
                for (std::uint32_t corner = 0u; corner < 16u; ++corner) {
                    const bool high_x         = (corner & 1u) != 0u;
                    const bool high_y         = (corner & 2u) != 0u;
                    const bool high_z         = (corner & 4u) != 0u;
                    const bool high_t         = (corner & 8u) != 0u;
                    const float weight        = (high_x ? frac_x : 1.0f - frac_x) * (high_y ? frac_y : 1.0f - frac_y) * (high_z ? frac_z : 1.0f - frac_z) * (high_t ? frac_t : 1.0f - frac_t);
                    const std::uint32_t index = hash_offsets[level] + grid_index4(hash_dense[level], hash_entries[level], rx, ry, rz, rt, grid_x + static_cast<std::uint32_t>(high_x), grid_y + static_cast<std::uint32_t>(high_y), grid_z + static_cast<std::uint32_t>(high_z), grid_t + static_cast<std::uint32_t>(high_t)) * HASH_FEATURES_PER_LEVEL;
                    atomicAdd(gradients + index + 0u, weight * encoded_grad[level * HASH_FEATURES_PER_LEVEL + 0u]);
                    atomicAdd(gradients + index + 1u, weight * encoded_grad[level * HASH_FEATURES_PER_LEVEL + 1u]);
                }
            }
        }

        __global__ void initialize_parameters_kernel(const std::uint32_t param_count, const std::uint32_t hash_param_count, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, float* const params, float* const gradients, float* const first_moments, float* const second_moments) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;
            Pcg32 rng{TRAIN_SEED};
            rng.advance(static_cast<std::uint64_t>(i) * 4ull);
            float value = 0.0f;
            if (i < hash_param_count) {
                value = rng.next_float() * 2e-4f - 1e-4f;
            } else if (i < w2_offset) {
                const float scale = sqrtf(6.0f / static_cast<float>(MLP_HIDDEN_WIDTH + HASH_INPUT_WIDTH));
                value             = rng.next_float() * 2.0f * scale - scale;
            } else if (i < color_offset) {
                const float scale = sqrtf(6.0f / static_cast<float>(MLP_HIDDEN_WIDTH + 1u));
                value             = rng.next_float() * 2.0f * scale - scale;
            }
            params[i]         = value;
            gradients[i]      = 0.0f;
            first_moments[i]  = 0.0f;
            second_moments[i] = 0.0f;
        }

        __global__ void train_density_kernel(const std::uint8_t* const train_pixels, const float* const train_cameras, const std::uint32_t train_view_count, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float* const world_to_sim, const float* const voxel_scale, const float near_plane, const float far_plane, const std::uint32_t samples_per_ray, const std::uint32_t rays_per_step, const std::uint32_t current_step, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, const std::uint8_t* const occupancy_bits, float* const params, float* const gradients, float* const loss_values, std::uint32_t* const skipped_sample_counter) {
            const std::uint32_t ray = threadIdx.x + blockIdx.x * blockDim.x;
            if (ray >= rays_per_step) return;

            Pcg32 rng{TRAIN_SEED};
            rng.advance((static_cast<std::uint64_t>(current_step) << 32ull) + static_cast<std::uint64_t>(ray) * 9973ull);
            const std::uint32_t view          = rng.next_uint() % train_view_count;
            const float pixel_x               = rng.next_float() * static_cast<float>(width - 1u);
            const float pixel_y               = rng.next_float() * static_cast<float>(height - 1u);
            const float raw_frame             = fminf(fmaxf(floorf(rng.next_float() * static_cast<float>(frame_count)) + rng.next_float() - 0.5f, 0.0f), static_cast<float>(frame_count - 1u));
            const float time_value            = frame_count > 1u ? clamp01(raw_frame / static_cast<float>(frame_count - 1u)) : 0.0f;
            const std::uint32_t occupancy_bin = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(time_value * static_cast<float>(TIME_OCCUPANCY_BINS - 1u) + 0.5f)), 0), static_cast<int>(TIME_OCCUPANCY_BINS - 1u)));

            float target_r = 0.0f;
            float target_g = 0.0f;
            float target_b = 0.0f;
            sample_video_rgb(train_pixels, view, raw_frame, pixel_x, pixel_y, frame_count, width, height, target_r, target_g, target_b);

            const float* const camera = train_cameras + static_cast<std::uint64_t>(view) * CAMERA_FLOATS;
            const float focal         = camera[16];
            const float camera_x      = (pixel_x - static_cast<float>(width) * 0.5f) / focal;
            const float camera_y      = -(pixel_y - static_cast<float>(height) * 0.5f) / focal;
            const float camera_z      = -1.0f;
            const float ray_origin_x  = camera[3];
            const float ray_origin_y  = camera[7];
            const float ray_origin_z  = camera[11];
            const float ray_dir_x     = camera[0] * camera_x + camera[1] * camera_y + camera[2] * camera_z;
            const float ray_dir_y     = camera[4] * camera_x + camera[5] * camera_y + camera[6] * camera_z;
            const float ray_dir_z     = camera[8] * camera_x + camera[9] * camera_y + camera[10] * camera_z;
            const float ray_length    = fmaxf(norm3df(ray_dir_x, ray_dir_y, ray_dir_z), 1e-8f);
            const float interval      = (far_plane - near_plane) / static_cast<float>(samples_per_ray);

            float optical_thickness = 0.0f;
            float encoded[HASH_INPUT_WIDTH];
            float hidden_pre[MLP_HIDDEN_WIDTH];
            float hidden[MLP_HIDDEN_WIDTH];
            std::uint32_t local_skipped_samples = 0u;
            for (std::uint32_t sample = 0u; sample < samples_per_ray; ++sample) {
                const float z           = near_plane + (static_cast<float>(sample) + 0.5f) * interval;
                const float px          = ray_origin_x + ray_dir_x * z;
                const float py          = ray_origin_y + ray_dir_y * z;
                const float pz          = ray_origin_z + ray_dir_z * z;
                const float sx_unscaled = world_to_sim[0] * px + world_to_sim[1] * py + world_to_sim[2] * pz + world_to_sim[3];
                const float sy_unscaled = world_to_sim[4] * px + world_to_sim[5] * py + world_to_sim[6] * pz + world_to_sim[7];
                const float sz_unscaled = world_to_sim[8] * px + world_to_sim[9] * py + world_to_sim[10] * pz + world_to_sim[11];
                const float sx          = sx_unscaled / voxel_scale[0];
                const float sy          = sy_unscaled / voxel_scale[1];
                const float sz          = sz_unscaled / voxel_scale[2];
                if (sx < BBOX_MIN_X || sx > BBOX_MAX_X || sy < BBOX_MIN_Y || sy > BBOX_MAX_Y || sz < BBOX_MIN_Z || sz > BBOX_MAX_Z) continue;
                const std::uint32_t occupancy_x    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sx - BBOX_MIN_X) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_X - BBOX_MIN_X))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_y    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sy - BBOX_MIN_Y) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Y - BBOX_MIN_Y))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_z    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sz - BBOX_MIN_Z) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Z - BBOX_MIN_Z))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_cell = occupancy_x + OCCUPANCY_GRID_SIZE * (occupancy_y + OCCUPANCY_GRID_SIZE * occupancy_z);
                const std::uint32_t occupancy_byte = occupancy_bin * (OCCUPANCY_GRID_CELLS / 8u) + occupancy_cell / 8u;
                const std::uint32_t occupancy_mask = 1u << (occupancy_cell & 7u);
                if ((static_cast<std::uint32_t>(occupancy_bits[occupancy_byte]) & occupancy_mask) == 0u) {
                    ++local_skipped_samples;
                    continue;
                }
                encode_hash4(px, py, pz, time_value, hash_offsets, hash_entries, hash_resolutions, hash_dense, params, encoded);
                const float sigma = forward_density(encoded, params, w1_offset, w2_offset, hidden_pre, hidden);
                optical_thickness += sigma * interval * ray_length;
            }
            if (local_skipped_samples != 0u) atomicAdd(skipped_sample_counter, local_skipped_samples);

            const float acc    = 1.0f - expf(-optical_thickness);
            const float color  = 0.6f + 0.4f * tanhf(params[color_offset]);
            const float pred_r = color * acc;
            const float pred_g = color * acc;
            const float pred_b = color * acc;
            const float diff_r = pred_r - target_r;
            const float diff_g = pred_g - target_g;
            const float diff_b = pred_b - target_b;
            loss_values[ray]   = (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b) * (1.0f / 3.0f);

            const float inv_rays     = 1.0f / static_cast<float>(rays_per_step);
            const float dloss_dacc   = (2.0f / 3.0f) * inv_rays * color * (diff_r + diff_g + diff_b);
            const float dcolor_draw  = 0.4f * (1.0f - tanhf(params[color_offset]) * tanhf(params[color_offset]));
            const float dloss_dcolor = (2.0f / 3.0f) * inv_rays * acc * (diff_r + diff_g + diff_b);
            atomicAdd(gradients + color_offset, dloss_dcolor * dcolor_draw);

            const float transmittance_end = expf(-optical_thickness);
            for (std::uint32_t sample = 0u; sample < samples_per_ray; ++sample) {
                const float z           = near_plane + (static_cast<float>(sample) + 0.5f) * interval;
                const float px          = ray_origin_x + ray_dir_x * z;
                const float py          = ray_origin_y + ray_dir_y * z;
                const float pz          = ray_origin_z + ray_dir_z * z;
                const float sx_unscaled = world_to_sim[0] * px + world_to_sim[1] * py + world_to_sim[2] * pz + world_to_sim[3];
                const float sy_unscaled = world_to_sim[4] * px + world_to_sim[5] * py + world_to_sim[6] * pz + world_to_sim[7];
                const float sz_unscaled = world_to_sim[8] * px + world_to_sim[9] * py + world_to_sim[10] * pz + world_to_sim[11];
                const float sx          = sx_unscaled / voxel_scale[0];
                const float sy          = sy_unscaled / voxel_scale[1];
                const float sz          = sz_unscaled / voxel_scale[2];
                if (sx < BBOX_MIN_X || sx > BBOX_MAX_X || sy < BBOX_MIN_Y || sy > BBOX_MAX_Y || sz < BBOX_MIN_Z || sz > BBOX_MAX_Z) continue;
                const std::uint32_t occupancy_x    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sx - BBOX_MIN_X) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_X - BBOX_MIN_X))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_y    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sy - BBOX_MIN_Y) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Y - BBOX_MIN_Y))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_z    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sz - BBOX_MIN_Z) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Z - BBOX_MIN_Z))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_cell = occupancy_x + OCCUPANCY_GRID_SIZE * (occupancy_y + OCCUPANCY_GRID_SIZE * occupancy_z);
                const std::uint32_t occupancy_byte = occupancy_bin * (OCCUPANCY_GRID_CELLS / 8u) + occupancy_cell / 8u;
                const std::uint32_t occupancy_mask = 1u << (occupancy_cell & 7u);
                if ((static_cast<std::uint32_t>(occupancy_bits[occupancy_byte]) & occupancy_mask) == 0u) continue;
                encode_hash4(px, py, pz, time_value, hash_offsets, hash_entries, hash_resolutions, hash_dense, params, encoded);
                const float sigma = forward_density(encoded, params, w1_offset, w2_offset, hidden_pre, hidden);
                if (sigma <= 0.0f) continue;
                const float dloss_dsigma = dloss_dacc * interval * ray_length * transmittance_end;
                float encoded_grad[HASH_INPUT_WIDTH];
                for (std::uint32_t i = 0u; i < HASH_INPUT_WIDTH; ++i) encoded_grad[i] = 0.0f;
                for (std::uint32_t row = 0u; row < MLP_HIDDEN_WIDTH; ++row) {
                    atomicAdd(gradients + w2_offset + row, dloss_dsigma * hidden[row]);
                    const float hidden_grad = dloss_dsigma * params[w2_offset + row] * (hidden_pre[row] > 0.0f ? 1.0f : 0.0f);
                    for (std::uint32_t column = 0u; column < HASH_INPUT_WIDTH; ++column) {
                        atomicAdd(gradients + w1_offset + row * HASH_INPUT_WIDTH + column, hidden_grad * encoded[column]);
                        encoded_grad[column] += hidden_grad * params[w1_offset + row * HASH_INPUT_WIDTH + column];
                    }
                }
                backward_hash4(px, py, pz, time_value, hash_offsets, hash_entries, hash_resolutions, hash_dense, encoded_grad, gradients);
            }
        }

        __global__ void update_time_occupancy_kernel(const std::uint32_t time_bin, const float* const sim_to_world, const float* const voxel_scale, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const float* const params, std::uint8_t* const occupancy_bits, float* const occupancy_values, std::uint32_t* const occupancy_counts) {
            const std::uint32_t byte_index               = threadIdx.x + blockIdx.x * blockDim.x;
            constexpr std::uint32_t occupancy_byte_count = OCCUPANCY_GRID_CELLS / 8u;
            if (byte_index >= occupancy_byte_count) return;

            const float time_value        = TIME_OCCUPANCY_BINS > 1u ? static_cast<float>(time_bin) / static_cast<float>(TIME_OCCUPANCY_BINS - 1u) : 0.0f;
            const std::uint32_t base_cell = byte_index * 8u;
            std::uint8_t byte_value       = 0u;
            std::uint32_t occupied_count  = 0u;
            float encoded[HASH_INPUT_WIDTH];
            float hidden_pre[MLP_HIDDEN_WIDTH];
            float hidden[MLP_HIDDEN_WIDTH];

            for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
                const std::uint32_t cell    = base_cell + bit;
                const std::uint32_t x_index = cell % OCCUPANCY_GRID_SIZE;
                const std::uint32_t y_index = (cell / OCCUPANCY_GRID_SIZE) % OCCUPANCY_GRID_SIZE;
                const std::uint32_t z_index = cell / (OCCUPANCY_GRID_SIZE * OCCUPANCY_GRID_SIZE);
                const float sim_x           = BBOX_MIN_X + (static_cast<float>(x_index) + 0.5f) * (BBOX_MAX_X - BBOX_MIN_X) / static_cast<float>(OCCUPANCY_GRID_SIZE);
                const float sim_y           = BBOX_MIN_Y + (static_cast<float>(y_index) + 0.5f) * (BBOX_MAX_Y - BBOX_MIN_Y) / static_cast<float>(OCCUPANCY_GRID_SIZE);
                const float sim_z           = BBOX_MIN_Z + (static_cast<float>(z_index) + 0.5f) * (BBOX_MAX_Z - BBOX_MIN_Z) / static_cast<float>(OCCUPANCY_GRID_SIZE);
                const float sim_scaled_x    = sim_x * voxel_scale[0];
                const float sim_scaled_y    = sim_y * voxel_scale[1];
                const float sim_scaled_z    = sim_z * voxel_scale[2];
                const float world_x         = sim_to_world[0] * sim_scaled_x + sim_to_world[1] * sim_scaled_y + sim_to_world[2] * sim_scaled_z + sim_to_world[3];
                const float world_y         = sim_to_world[4] * sim_scaled_x + sim_to_world[5] * sim_scaled_y + sim_to_world[6] * sim_scaled_z + sim_to_world[7];
                const float world_z         = sim_to_world[8] * sim_scaled_x + sim_to_world[9] * sim_scaled_y + sim_to_world[10] * sim_scaled_z + sim_to_world[11];
                encode_hash4(world_x, world_y, world_z, time_value, hash_offsets, hash_entries, hash_resolutions, hash_dense, params, encoded);
                const float sigma               = forward_density(encoded, params, w1_offset, w2_offset, hidden_pre, hidden);
                const std::uint32_t value_index = time_bin * OCCUPANCY_GRID_CELLS + cell;
                const float previous            = occupancy_values[value_index];
                const float updated             = fmaxf(previous * OCCUPANCY_EMA_DECAY, sigma);
                occupancy_values[value_index]   = updated;
                if (updated > OCCUPANCY_THRESHOLD) {
                    byte_value = static_cast<std::uint8_t>(byte_value | static_cast<std::uint8_t>(1u << bit));
                    ++occupied_count;
                }
            }

            occupancy_bits[time_bin * occupancy_byte_count + byte_index] = byte_value;
            if (occupied_count != 0u) atomicAdd(occupancy_counts + time_bin, occupied_count);
        }

        __global__ void radam_kernel(const std::uint32_t param_count, const std::uint32_t hash_param_count, const std::uint32_t current_step, const float learning_rate, float* const params, const float* const gradients, float* const first_moments, float* const second_moments) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;
            float gradient = gradients[i];
            if (i >= hash_param_count) gradient += OPTIMIZER_WEIGHT_DECAY * params[i];
            const float beta1_power = powf(OPTIMIZER_BETA1, static_cast<float>(current_step + 1u));
            const float beta2_power = powf(OPTIMIZER_BETA2, static_cast<float>(current_step + 1u));
            const float first = first_moments[i] = OPTIMIZER_BETA1 * first_moments[i] + (1.0f - OPTIMIZER_BETA1) * gradient;
            const float second = second_moments[i] = OPTIMIZER_BETA2 * second_moments[i] + (1.0f - OPTIMIZER_BETA2) * gradient * gradient;
            const float rho_inf                    = 2.0f / (1.0f - OPTIMIZER_BETA2) - 1.0f;
            const float rho                        = rho_inf - 2.0f * static_cast<float>(current_step + 1u) * beta2_power / (1.0f - beta2_power);
            if (rho > 5.0f) {
                const float rect = sqrtf(((rho - 4.0f) * (rho - 2.0f) * rho_inf) / ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho));
                params[i] -= learning_rate * rect * (first / (1.0f - beta1_power)) / (sqrtf(second / (1.0f - beta2_power)) + OPTIMIZER_EPSILON);
            } else {
                params[i] -= learning_rate * first / (1.0f - beta1_power);
            }
        }

        __global__ void evaluate_frame_kernel(const std::uint8_t* const test_pixels, const float* const test_cameras, const std::uint32_t test_view_index, const std::uint32_t frame_index, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float* const world_to_sim, const float* const voxel_scale, const float near_plane, const float far_plane, const std::uint32_t samples_per_ray, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, const std::uint8_t* const occupancy_bits, const float* const params, double* const loss_sum, std::uint8_t* const comparison_pixels) {
            const std::uint32_t pixel       = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t pixel_count = width * height;
            if (pixel >= pixel_count) return;
            const std::uint32_t x             = pixel % width;
            const std::uint32_t y             = pixel / width;
            const float* const camera         = test_cameras + static_cast<std::uint64_t>(test_view_index) * CAMERA_FLOATS;
            const float focal                 = camera[16];
            const float camera_x              = (static_cast<float>(x) - static_cast<float>(width) * 0.5f) / focal;
            const float camera_y              = -(static_cast<float>(y) - static_cast<float>(height) * 0.5f) / focal;
            const float camera_z              = -1.0f;
            const float ray_origin_x          = camera[3];
            const float ray_origin_y          = camera[7];
            const float ray_origin_z          = camera[11];
            const float ray_dir_x             = camera[0] * camera_x + camera[1] * camera_y + camera[2] * camera_z;
            const float ray_dir_y             = camera[4] * camera_x + camera[5] * camera_y + camera[6] * camera_z;
            const float ray_dir_z             = camera[8] * camera_x + camera[9] * camera_y + camera[10] * camera_z;
            const float ray_length            = fmaxf(norm3df(ray_dir_x, ray_dir_y, ray_dir_z), 1e-8f);
            const float interval              = (far_plane - near_plane) / static_cast<float>(samples_per_ray);
            const float time_value            = frame_count > 1u ? static_cast<float>(frame_index) / static_cast<float>(frame_count - 1u) : 0.0f;
            const std::uint32_t occupancy_bin = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf(time_value * static_cast<float>(TIME_OCCUPANCY_BINS - 1u) + 0.5f)), 0), static_cast<int>(TIME_OCCUPANCY_BINS - 1u)));

            float encoded[HASH_INPUT_WIDTH];
            float hidden_pre[MLP_HIDDEN_WIDTH];
            float hidden[MLP_HIDDEN_WIDTH];
            float optical_thickness = 0.0f;
            for (std::uint32_t sample = 0u; sample < samples_per_ray; ++sample) {
                const float z           = near_plane + (static_cast<float>(sample) + 0.5f) * interval;
                const float px          = ray_origin_x + ray_dir_x * z;
                const float py          = ray_origin_y + ray_dir_y * z;
                const float pz          = ray_origin_z + ray_dir_z * z;
                const float sx_unscaled = world_to_sim[0] * px + world_to_sim[1] * py + world_to_sim[2] * pz + world_to_sim[3];
                const float sy_unscaled = world_to_sim[4] * px + world_to_sim[5] * py + world_to_sim[6] * pz + world_to_sim[7];
                const float sz_unscaled = world_to_sim[8] * px + world_to_sim[9] * py + world_to_sim[10] * pz + world_to_sim[11];
                const float sx          = sx_unscaled / voxel_scale[0];
                const float sy          = sy_unscaled / voxel_scale[1];
                const float sz          = sz_unscaled / voxel_scale[2];
                if (sx < BBOX_MIN_X || sx > BBOX_MAX_X || sy < BBOX_MIN_Y || sy > BBOX_MAX_Y || sz < BBOX_MIN_Z || sz > BBOX_MAX_Z) continue;
                const std::uint32_t occupancy_x    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sx - BBOX_MIN_X) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_X - BBOX_MIN_X))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_y    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sy - BBOX_MIN_Y) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Y - BBOX_MIN_Y))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_z    = static_cast<std::uint32_t>(min(max(static_cast<int>(floorf((sz - BBOX_MIN_Z) * static_cast<float>(OCCUPANCY_GRID_SIZE) / (BBOX_MAX_Z - BBOX_MIN_Z))), 0), static_cast<int>(OCCUPANCY_GRID_SIZE - 1u)));
                const std::uint32_t occupancy_cell = occupancy_x + OCCUPANCY_GRID_SIZE * (occupancy_y + OCCUPANCY_GRID_SIZE * occupancy_z);
                const std::uint32_t occupancy_byte = occupancy_bin * (OCCUPANCY_GRID_CELLS / 8u) + occupancy_cell / 8u;
                const std::uint32_t occupancy_mask = 1u << (occupancy_cell & 7u);
                if ((static_cast<std::uint32_t>(occupancy_bits[occupancy_byte]) & occupancy_mask) == 0u) continue;
                encode_hash4(px, py, pz, time_value, hash_offsets, hash_entries, hash_resolutions, hash_dense, params, encoded);
                optical_thickness += forward_density(encoded, params, w1_offset, w2_offset, hidden_pre, hidden) * interval * ray_length;
            }

            const float acc      = 1.0f - expf(-optical_thickness);
            const float color    = 0.6f + 0.4f * tanhf(params[color_offset]);
            const float pred     = clamp01(color * acc);
            const float target_r = sample_video_channel(test_pixels, test_view_index, frame_index, frame_count, width, height, x, y, 0u);
            const float target_g = sample_video_channel(test_pixels, test_view_index, frame_index, frame_count, width, height, x, y, 1u);
            const float target_b = sample_video_channel(test_pixels, test_view_index, frame_index, frame_count, width, height, x, y, 2u);
            const double diff_r  = static_cast<double>(pred) - static_cast<double>(target_r);
            const double diff_g  = static_cast<double>(pred) - static_cast<double>(target_g);
            const double diff_b  = static_cast<double>(pred) - static_cast<double>(target_b);
            atomicAdd(loss_sum, diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);

            const std::uint64_t row_offset          = static_cast<std::uint64_t>(y) * width * 2ull * 3ull;
            const std::uint64_t target_offset       = row_offset + static_cast<std::uint64_t>(x) * 3ull;
            const std::uint64_t render_offset       = row_offset + static_cast<std::uint64_t>(width + x) * 3ull;
            comparison_pixels[target_offset + 0ull] = static_cast<std::uint8_t>(clamp01(target_r) * 255.0f + 0.5f);
            comparison_pixels[target_offset + 1ull] = static_cast<std::uint8_t>(clamp01(target_g) * 255.0f + 0.5f);
            comparison_pixels[target_offset + 2ull] = static_cast<std::uint8_t>(clamp01(target_b) * 255.0f + 0.5f);
            comparison_pixels[render_offset + 0ull] = static_cast<std::uint8_t>(pred * 255.0f + 0.5f);
            comparison_pixels[render_offset + 1ull] = static_cast<std::uint8_t>(pred * 255.0f + 0.5f);
            comparison_pixels[render_offset + 2ull] = static_cast<std::uint8_t>(pred * 255.0f + 0.5f);
        }
    } // namespace

    void upload_bytes(const std::uint8_t* const data, const std::size_t byte_count, const std::uint8_t*& out_data) {
        if (data == nullptr || byte_count == 0) throw std::runtime_error{"invalid byte upload input."};
        void* ptr = nullptr;
        if (const cudaError_t status = cudaMalloc(&ptr, byte_count); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc bytes failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(ptr, data, byte_count, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy bytes failed: "} + cudaGetErrorString(status)};
        out_data = static_cast<const std::uint8_t*>(ptr);
    }

    void upload_floats(const float* const data, const std::size_t value_count, const float*& out_data) {
        if (data == nullptr || value_count == 0) throw std::runtime_error{"invalid float upload input."};
        void* ptr = nullptr;
        if (const cudaError_t status = cudaMalloc(&ptr, value_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc floats failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(ptr, data, value_count * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy floats failed: "} + cudaGetErrorString(status)};
        out_data = static_cast<const float*>(ptr);
    }

    void upload_uint32s(const std::uint32_t* const data, const std::size_t value_count, const std::uint32_t*& out_data) {
        if (data == nullptr || value_count == 0) throw std::runtime_error{"invalid uint32 upload input."};
        void* ptr = nullptr;
        if (const cudaError_t status = cudaMalloc(&ptr, value_count * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc uint32s failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(ptr, data, value_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy uint32s failed: "} + cudaGetErrorString(status)};
        out_data = static_cast<const std::uint32_t*>(ptr);
    }

    void allocate_float_buffer(const std::size_t value_count, float*& out_data) {
        if (value_count == 0) throw std::runtime_error{"float allocation size must be non-zero."};
        if (const cudaError_t status = cudaMalloc(&out_data, value_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc float buffer failed: "} + cudaGetErrorString(status)};
    }

    void allocate_double_buffer(const std::size_t value_count, double*& out_data) {
        if (value_count == 0) throw std::runtime_error{"double allocation size must be non-zero."};
        if (const cudaError_t status = cudaMalloc(&out_data, value_count * sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc double buffer failed: "} + cudaGetErrorString(status)};
    }

    void allocate_byte_buffer(const std::size_t byte_count, std::uint8_t*& out_data) {
        if (byte_count == 0) throw std::runtime_error{"byte allocation size must be non-zero."};
        if (const cudaError_t status = cudaMalloc(&out_data, byte_count); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc byte buffer failed: "} + cudaGetErrorString(status)};
    }

    void allocate_uint32_buffer(const std::size_t value_count, std::uint32_t*& out_data) {
        if (value_count == 0) throw std::runtime_error{"uint32 allocation size must be non-zero."};
        if (const cudaError_t status = cudaMalloc(&out_data, value_count * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc uint32 buffer failed: "} + cudaGetErrorString(status)};
    }

    void initialize_parameters(const std::uint32_t param_count, const std::uint32_t hash_param_count, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, float* const params, float* const gradients, float* const first_moments, float* const second_moments) {
        if (param_count == 0u || params == nullptr || gradients == nullptr || first_moments == nullptr || second_moments == nullptr) throw std::runtime_error{"invalid parameter initialization input."};
        initialize_parameters_kernel<<<(param_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(param_count, hash_param_count, w1_offset, w2_offset, color_offset, params, gradients, first_moments, second_moments);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"initialize_parameters_kernel failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"initialize parameter synchronization failed: "} + cudaGetErrorString(status)};
    }

    void initialize_occupancy(std::uint8_t* const occupancy_bits, float* const occupancy_values, std::uint32_t* const occupancy_counts) {
        if (occupancy_bits == nullptr || occupancy_values == nullptr || occupancy_counts == nullptr) throw std::runtime_error{"invalid occupancy initialization input."};
        if (const cudaError_t status = cudaMemset(occupancy_bits, 0xFF, static_cast<std::size_t>(TIME_OCCUPANCY_BINS) * OCCUPANCY_GRID_CELLS / 8); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset occupancy bits failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(occupancy_values, 0, static_cast<std::size_t>(TIME_OCCUPANCY_BINS) * OCCUPANCY_GRID_CELLS * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset occupancy values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(occupancy_counts, 0, static_cast<std::size_t>(TIME_OCCUPANCY_BINS) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset occupancy counts failed: "} + cudaGetErrorString(status)};
    }

    void update_time_occupancy(const std::uint32_t time_bin, const float* const sim_to_world, const float* const voxel_scale, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const float* const params, std::uint8_t* const occupancy_bits, float* const occupancy_values, std::uint32_t* const occupancy_counts) {
        if (time_bin >= TIME_OCCUPANCY_BINS || sim_to_world == nullptr || voxel_scale == nullptr || hash_offsets == nullptr || hash_entries == nullptr || hash_resolutions == nullptr || hash_dense == nullptr || params == nullptr || occupancy_bits == nullptr || occupancy_values == nullptr || occupancy_counts == nullptr) throw std::runtime_error{"invalid occupancy update input."};
        if (const cudaError_t status = cudaMemset(occupancy_counts + time_bin, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset occupancy count failed: "} + cudaGetErrorString(status)};
        constexpr std::uint32_t occupancy_byte_count = OCCUPANCY_GRID_CELLS / 8u;
        update_time_occupancy_kernel<<<(occupancy_byte_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(time_bin, sim_to_world, voxel_scale, hash_offsets, hash_entries, hash_resolutions, hash_dense, w1_offset, w2_offset, params, occupancy_bits, occupancy_values, occupancy_counts);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"update_time_occupancy_kernel failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"occupancy update synchronization failed: "} + cudaGetErrorString(status)};
    }

    void run_training_step(const std::uint8_t* const train_pixels, const float* const train_cameras, const std::uint32_t train_view_count, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float* const world_to_sim, const float* const voxel_scale, const float near_plane, const float far_plane, const std::uint32_t samples_per_ray, const std::uint32_t rays_per_step, const std::uint32_t current_step, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t hash_param_count, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, const std::uint8_t* const occupancy_bits, float* const params, float* const gradients, float* const loss_values, std::uint32_t* const skipped_sample_counter) {
        if (train_pixels == nullptr || train_cameras == nullptr || train_view_count == 0u || frame_count == 0u || width == 0u || height == 0u || world_to_sim == nullptr || voxel_scale == nullptr || near_plane <= 0.0f || far_plane <= near_plane || samples_per_ray == 0u || rays_per_step == 0u || hash_offsets == nullptr || hash_entries == nullptr || hash_resolutions == nullptr || hash_dense == nullptr || hash_param_count == 0u || occupancy_bits == nullptr || params == nullptr || gradients == nullptr || loss_values == nullptr || skipped_sample_counter == nullptr) throw std::runtime_error{"invalid training step input."};
        if (const cudaError_t status = cudaMemset(gradients, 0, static_cast<std::size_t>(color_offset + 1u) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(loss_values, 0, static_cast<std::size_t>(rays_per_step) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loss values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(skipped_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset skipped sample counter failed: "} + cudaGetErrorString(status)};
        train_density_kernel<<<(rays_per_step + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(train_pixels, train_cameras, train_view_count, frame_count, width, height, world_to_sim, voxel_scale, near_plane, far_plane, samples_per_ray, rays_per_step, current_step, hash_offsets, hash_entries, hash_resolutions, hash_dense, w1_offset, w2_offset, color_offset, occupancy_bits, params, gradients, loss_values, skipped_sample_counter);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"train_density_kernel failed: "} + cudaGetErrorString(status)};
    }

    void step_radam(const std::uint32_t param_count, const std::uint32_t hash_param_count, const std::uint32_t current_step, const float learning_rate, float* const params, float* const gradients, float* const first_moments, float* const second_moments) {
        if (param_count == 0u || hash_param_count > param_count || !std::isfinite(learning_rate) || learning_rate <= 0.0f || params == nullptr || gradients == nullptr || first_moments == nullptr || second_moments == nullptr) throw std::runtime_error{"invalid RAdam input."};
        radam_kernel<<<(param_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(param_count, hash_param_count, current_step, learning_rate, params, gradients, first_moments, second_moments);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"radam_kernel failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"RAdam synchronization failed: "} + cudaGetErrorString(status)};
    }

    void evaluate_test_frame(const std::uint8_t* const test_pixels, const float* const test_cameras, const std::uint32_t test_view_index, const std::uint32_t frame_index, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float* const world_to_sim, const float* const voxel_scale, const float near_plane, const float far_plane, const std::uint32_t samples_per_ray, const std::uint32_t* const hash_offsets, const std::uint32_t* const hash_entries, const std::uint32_t* const hash_resolutions, const std::uint32_t* const hash_dense, const std::uint32_t w1_offset, const std::uint32_t w2_offset, const std::uint32_t color_offset, const std::uint8_t* const occupancy_bits, const float* const params, double* const loss_sum, std::uint8_t* const comparison_pixels) {
        if (test_pixels == nullptr || test_cameras == nullptr || frame_index >= frame_count || frame_count == 0u || width == 0u || height == 0u || world_to_sim == nullptr || voxel_scale == nullptr || near_plane <= 0.0f || far_plane <= near_plane || samples_per_ray == 0u || hash_offsets == nullptr || hash_entries == nullptr || hash_resolutions == nullptr || hash_dense == nullptr || occupancy_bits == nullptr || params == nullptr || loss_sum == nullptr || comparison_pixels == nullptr) throw std::runtime_error{"invalid test evaluation input."};
        if (const cudaError_t status = cudaMemset(loss_sum, 0, sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset test loss failed: "} + cudaGetErrorString(status)};
        const std::uint32_t pixel_count = width * height;
        evaluate_frame_kernel<<<(pixel_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(test_pixels, test_cameras, test_view_index, frame_index, frame_count, width, height, world_to_sim, voxel_scale, near_plane, far_plane, samples_per_ray, hash_offsets, hash_entries, hash_resolutions, hash_dense, w1_offset, w2_offset, color_offset, occupancy_bits, params, loss_sum, comparison_pixels);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"evaluate_frame_kernel failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"test evaluation synchronization failed: "} + cudaGetErrorString(status)};
    }

    void download_floats(const float* const data, const std::size_t value_count, float* const out_data) {
        if (data == nullptr || out_data == nullptr || value_count == 0) throw std::runtime_error{"invalid float download input."};
        if (const cudaError_t status = cudaMemcpy(out_data, data, value_count * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy float download failed: "} + cudaGetErrorString(status)};
    }

    void upload_parameters(const std::uint32_t param_count, const float* const data, float* const params, float* const gradients, float* const first_moments, float* const second_moments) {
        if (param_count == 0u || data == nullptr || params == nullptr || gradients == nullptr || first_moments == nullptr || second_moments == nullptr) throw std::runtime_error{"invalid parameter upload input."};
        if (const cudaError_t status = cudaMemcpy(params, data, static_cast<std::size_t>(param_count) * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy parameter upload failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(gradients, 0, static_cast<std::size_t>(param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(first_moments, 0, static_cast<std::size_t>(param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(second_moments, 0, static_cast<std::size_t>(param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded second moments failed: "} + cudaGetErrorString(status)};
    }

    void download_double(const double* const data, double& out_value) {
        if (data == nullptr) throw std::runtime_error{"invalid double download input."};
        if (const cudaError_t status = cudaMemcpy(&out_value, data, sizeof(double), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy double download failed: "} + cudaGetErrorString(status)};
    }

    void download_uint32(const std::uint32_t* const data, std::uint32_t& out_value) {
        if (data == nullptr) throw std::runtime_error{"invalid uint32 download input."};
        if (const cudaError_t status = cudaMemcpy(&out_value, data, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy uint32 download failed: "} + cudaGetErrorString(status)};
    }

    void download_bytes(const std::uint8_t* const data, const std::size_t byte_count, std::uint8_t* const out_data) {
        if (data == nullptr || out_data == nullptr || byte_count == 0) throw std::runtime_error{"invalid byte download input."};
        if (const cudaError_t status = cudaMemcpy(out_data, data, byte_count, cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy byte download failed: "} + cudaGetErrorString(status)};
    }
} // namespace hyfluid::cuda
