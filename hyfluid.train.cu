#include "hyfluid.train.config.h"
#include "hyfluid.train.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda/std/algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

        struct CublasLtMatmulResources final {
            cublasLtMatmulDesc_t operation_desc   = nullptr;
            cublasLtMatrixLayout_t a_desc         = nullptr;
            cublasLtMatrixLayout_t b_desc         = nullptr;
            cublasLtMatrixLayout_t d_desc         = nullptr;
            cublasLtMatmulPreference_t preference = nullptr;

            ~CublasLtMatmulResources() noexcept {
                if (this->preference != nullptr) cublasLtMatmulPreferenceDestroy(this->preference);
                if (this->d_desc != nullptr) cublasLtMatrixLayoutDestroy(this->d_desc);
                if (this->b_desc != nullptr) cublasLtMatrixLayoutDestroy(this->b_desc);
                if (this->a_desc != nullptr) cublasLtMatrixLayoutDestroy(this->a_desc);
                if (this->operation_desc != nullptr) cublasLtMatmulDescDestroy(this->operation_desc);
            }
        };

        __host__ void cublaslt_matmul_row_major(cublasLtHandle_t cublaslt, const cublasOperation_t transa, const cublasOperation_t transb, const cudaDataType_t a_type, const cudaDataType_t b_type, const cudaDataType_t d_type, const void* a, const int a_rows, const int a_cols, const void* b, const int b_rows, const int b_cols, void* d, const int d_rows, const int d_cols, const float alpha, const float beta, std::uint8_t* workspace) {
            CublasLtMatmulResources lt;
            constexpr cublasLtOrder_t row_order       = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = train::config::cublaslt_workspace_bytes;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(lt.operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescSetAttribute transa failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(lt.operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescSetAttribute transb failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, a_type, a_rows, a_cols, a_cols); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, b_type, b_rows, b_cols, b_cols); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, d_type, d_rows, d_cols, d_cols); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt returned no supported matmul algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, a, lt.a_desc, b, lt.b_desc, &beta, d, lt.d_desc, d, lt.d_desc, &heuristic.algo, workspace, train::config::cublaslt_workspace_bytes, nullptr); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmul failed: "} + cublasGetStatusString(status)};
        }

        __host__ __device__ std::uint32_t hash4_level_resolution(const std::uint32_t level) {
            switch (level) {
            case 0u: return 16u;
            case 1u: return 21u;
            case 2u: return 26u;
            case 3u: return 32u;
            case 4u: return 41u;
            case 5u: return 51u;
            case 6u: return 65u;
            case 7u: return 81u;
            case 8u: return 102u;
            case 9u: return 129u;
            case 10u: return 162u;
            case 11u: return 204u;
            case 12u: return 257u;
            case 13u: return 323u;
            case 14u: return 407u;
            default: return 512u;
            }
        }

        __host__ __device__ std::uint32_t hash4_level_offset(const std::uint32_t level) {
            switch (level) {
            case 0u: return train::config::hash4_offset_0;
            case 1u: return train::config::hash4_offset_1;
            case 2u: return train::config::hash4_offset_2;
            case 3u: return train::config::hash4_offset_3;
            case 4u: return train::config::hash4_offset_4;
            case 5u: return train::config::hash4_offset_5;
            case 6u: return train::config::hash4_offset_6;
            case 7u: return train::config::hash4_offset_7;
            case 8u: return train::config::hash4_offset_8;
            case 9u: return train::config::hash4_offset_9;
            case 10u: return train::config::hash4_offset_10;
            case 11u: return train::config::hash4_offset_11;
            case 12u: return train::config::hash4_offset_12;
            case 13u: return train::config::hash4_offset_13;
            case 14u: return train::config::hash4_offset_14;
            case 15u: return train::config::hash4_offset_15;
            default: return train::config::hash4_offset_16;
            }
        }

        __host__ __device__ std::uint32_t hash4_level_entry_count(const std::uint32_t level) {
            return (hash4_level_offset(level + 1u) - hash4_level_offset(level)) / train::config::hash4_features_per_level;
        }

        __device__ std::uint32_t fast_hash4(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t t) {
            std::uint32_t result = 0u;
            result ^= x * 1u;
            result ^= y * 2654435761u;
            result ^= z * 805459861u;
            result ^= t * 3674653429u;
            return result;
        }

        __device__ std::uint32_t hash4_entry_index(const std::uint32_t level, const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t t) {
            const std::uint32_t resolution    = hash4_level_resolution(level);
            const std::uint32_t entries       = hash4_level_entry_count(level);
            const std::uint64_t axis          = static_cast<std::uint64_t>(resolution) + 1ull;
            const std::uint64_t dense_entries = axis * axis * axis * axis;
            if (dense_entries <= static_cast<std::uint64_t>(entries)) return static_cast<std::uint32_t>(static_cast<std::uint64_t>(x) + axis * static_cast<std::uint64_t>(y) + axis * axis * static_cast<std::uint64_t>(z) + axis * axis * axis * static_cast<std::uint64_t>(t));
            return fast_hash4(x, y, z, t) % entries;
        }

        __device__ bool unit_aabb_contains(const float3 pos) {
            return pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f && pos.z >= 0.0f && pos.z <= 1.0f;
        }

        __device__ bool intersect_unit_aabb(const float3 origin, const float3 direction, float& out_tmin, float& out_tmax) {
            const float3 inv_dir = {1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z};
            const float3 t0      = {-origin.x * inv_dir.x, -origin.y * inv_dir.y, -origin.z * inv_dir.z};
            const float3 t1      = {(1.0f - origin.x) * inv_dir.x, (1.0f - origin.y) * inv_dir.y, (1.0f - origin.z) * inv_dir.z};
            const float tx_min   = fminf(t0.x, t1.x);
            const float tx_max   = fmaxf(t0.x, t1.x);
            const float ty_min   = fminf(t0.y, t1.y);
            const float ty_max   = fmaxf(t0.y, t1.y);
            const float tz_min   = fminf(t0.z, t1.z);
            const float tz_max   = fmaxf(t0.z, t1.z);
            const float tmin     = fmaxf(fmaxf(tx_min, ty_min), tz_min);
            const float tmax     = fminf(fminf(tx_max, ty_max), tz_max);
            out_tmin             = fmaxf(tmin, 0.0f);
            out_tmax             = tmax;
            return out_tmax >= out_tmin;
        }

        __device__ float field_metric_per_unit(const float* field_to_world_linear, const float3 direction) {
            const float x = field_to_world_linear[0u] * direction.x + field_to_world_linear[1u] * direction.y + field_to_world_linear[2u] * direction.z;
            const float y = field_to_world_linear[3u] * direction.x + field_to_world_linear[4u] * direction.y + field_to_world_linear[5u] * direction.z;
            const float z = field_to_world_linear[6u] * direction.x + field_to_world_linear[7u] * direction.y + field_to_world_linear[8u] * direction.z;
            return norm3df(x, y, z);
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

        __device__ float advance_to_next_density_voxel(const float t, const float3 pos, const float3 direction, const float3 inv_direction, const float step_size) {
            constexpr auto scale = static_cast<float>(train::config::nerf_grid_size);
            const float3 p       = {(pos.x - 0.5f) * scale, (pos.y - 0.5f) * scale, (pos.z - 0.5f) * scale};
            const float tx       = (floorf(p.x + 0.5f + 0.5f * copysignf(1.0f, direction.x)) - p.x) * inv_direction.x;
            const float ty       = (floorf(p.y + 0.5f + 0.5f * copysignf(1.0f, direction.y)) - p.y) * inv_direction.y;
            const float tz       = (floorf(p.z + 0.5f + 0.5f * copysignf(1.0f, direction.z)) - p.z) * inv_direction.z;
            const float t_target = t + fmaxf(fminf(fminf(tx, ty), tz) / scale, 0.0f);
            return t + ceilf(fmaxf((t_target - t) / step_size, 0.5f)) * step_size;
        }

        __device__ void replay_training_ray_rng(const std::uint32_t ray_index, const std::uint32_t current_step, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, std::uint32_t& out_view_index, std::uint32_t& out_floor_time_index, std::uint32_t& out_ceil_time_index, float& out_time_residual, std::uint32_t& out_pixel_x, std::uint32_t& out_pixel_y, float& out_ray_phase) {
            Pcg32 rng{train::config::train_seed};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(ray_index) * train::config::max_random_samples_per_ray);

            out_view_index                      = rng.next_uint() % view_count;
            const std::uint32_t base_time_index = time_count == 1u ? 0u : rng.next_uint() % time_count;
            const auto max_time_index           = static_cast<float>(time_count - 1u);
            const float fractional_time_index   = time_count == 1u ? 0.0f : fminf(fmaxf(static_cast<float>(base_time_index) + rng.next_float() - 0.5f, 0.0f), max_time_index);
            out_floor_time_index                = static_cast<std::uint32_t>(floorf(fractional_time_index));
            out_ceil_time_index                 = time_count == 1u ? 0u : ::cuda::std::min(out_floor_time_index + 1u, time_count - 1u);
            out_time_residual                   = fractional_time_index - static_cast<float>(out_floor_time_index);

            const float u   = rng.next_float();
            const float v   = rng.next_float();
            out_pixel_x     = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
            out_pixel_y     = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
            out_ray_phase    = rng.next_float();
        }

        __global__ void generate_training_samples_kernel(const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint32_t current_step, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, const float* __restrict__ camera, const float* __restrict__ intrinsics, const std::uint32_t* __restrict__ frame_indices, const float* __restrict__ field_to_world_linear, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ ray_indices_out, float* __restrict__ rays_out, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= rays_per_batch) return;

            std::uint32_t view_index       = 0u;
            std::uint32_t floor_time_index = 0u;
            std::uint32_t ceil_time_index  = 0u;
            float time_residual            = 0.0f;
            std::uint32_t pixel_x          = 0u;
            std::uint32_t pixel_y          = 0u;
            float ray_phase                = 0.0f;
            replay_training_ray_rng(i, current_step, view_count, time_count, width, height, view_index, floor_time_index, ceil_time_index, time_residual, pixel_x, pixel_y, ray_phase);

            const auto max_time_index             = static_cast<float>(time_count - 1u);
            const float normalized_time           = time_count == 1u ? 0.0f : (static_cast<float>(floor_time_index) + time_residual) / max_time_index;
            const std::uint32_t floor_frame_index = frame_indices[view_index * time_count + floor_time_index];
            const float* frame_camera             = camera + static_cast<std::uint64_t>(floor_frame_index) * 12u;
            const float* frame_intrinsics         = intrinsics + static_cast<std::uint64_t>(floor_frame_index) * 4u;
            const float sample_x                  = static_cast<float>(pixel_x) + 0.5f;
            const float sample_y                  = static_cast<float>(pixel_y) + 0.5f;
            const float ray_x                     = (sample_x - frame_intrinsics[2u]) / frame_intrinsics[0u];
            const float ray_y                     = (sample_y - frame_intrinsics[3u]) / frame_intrinsics[1u];
            const float3 camera_x                 = {frame_camera[0], frame_camera[1], frame_camera[2]};
            const float3 camera_y                 = {frame_camera[3], frame_camera[4], frame_camera[5]};
            const float3 camera_z                 = {frame_camera[6], frame_camera[7], frame_camera[8]};
            const float3 ray_origin               = {frame_camera[9], frame_camera[10], frame_camera[11]};
            float3 ray_direction                  = {
                camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
            };
            if (ray_direction.x == 0.0f && ray_direction.y == 0.0f && ray_direction.z == 0.0f) ray_direction = camera_z;

            const float direction_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
            if (direction_length == 0.0f) return;
            const float3 ray_direction_normalized = {ray_direction.x / direction_length, ray_direction.y / direction_length, ray_direction.z / direction_length};
            const float metric_per_field_unit     = field_metric_per_unit(field_to_world_linear, ray_direction_normalized);
            if (!isfinite(metric_per_field_unit) || metric_per_field_unit <= 0.0f) return;

            const std::uint32_t ray_index = atomicAdd(ray_counter, 1u);
            ray_indices_out[ray_index]    = i;

            float* ray_out = rays_out + static_cast<std::uint64_t>(ray_index) * train::config::ray_floats;
            ray_out[0u]    = ray_origin.x;
            ray_out[1u]    = ray_origin.y;
            ray_out[2u]    = ray_origin.z;
            ray_out[3u]    = ray_direction_normalized.x;
            ray_out[4u]    = ray_direction_normalized.y;
            ray_out[5u]    = ray_direction_normalized.z;

            numsteps_out[ray_index * 2u + 0u] = 0u;
            numsteps_out[ray_index * 2u + 1u] = 0u;

            float tmin = 0.0f;
            float tmax = 0.0f;
            if (!intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin, tmax)) return;

            constexpr float dt         = train::config::training_ray_stepsize;
            const float dt_metric      = dt * metric_per_field_unit;
            const float start_t        = fmaf(ray_phase, dt, tmin);
            const float3 inv_direction = {1.0f / ray_direction_normalized.x, 1.0f / ray_direction_normalized.y, 1.0f / ray_direction_normalized.z};

            std::uint32_t numsteps = 0u;
            float t                = start_t;
            float3 pos             = {};

            while (numsteps < train::config::training_ray_steps && t <= tmax) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;
                if (is_occupancy_grid_cell_occupied(pos, occupancy)) {
                    ++numsteps;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction, dt);
                }
            }

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base >= sample_limit) return;
            const std::uint32_t stored_numsteps = ::cuda::std::min(numsteps, sample_limit - base);
            if (stored_numsteps == 0u) return;

            numsteps_out[ray_index * 2u + 0u] = stored_numsteps;
            numsteps_out[ray_index * 2u + 1u] = base;

            t               = start_t;
            std::uint32_t j = 0u;
            while (j < stored_numsteps && t <= tmax) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;
                if (is_occupancy_grid_cell_occupied(pos, occupancy)) {
                    float* coord = coords_out + static_cast<std::uint64_t>(base + j) * train::config::sample_coord_floats;
                    coord[0u]    = pos.x;
                    coord[1u]    = pos.y;
                    coord[2u]    = pos.z;
                    coord[3u]    = normalized_time;
                    coord[4u]    = dt_metric;
                    ++j;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction, dt);
                }
            }
        }

        __global__ void initialize_trainable_parameters_kernel(float* __restrict__ params_full_precision, __half* __restrict__ params) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= train::config::network_parameter_layout.total_param_count) return;

            float value = 0.0f;
            Pcg32 rng{0x9e3779b97f4a7c15ull};
            rng.advance(i);
            if (i < train::config::network_parameter_layout.mlp_output_weight_offset) {
                const float scale = 1.0f / sqrtf(static_cast<float>(train::config::mlp_input_width));
                value             = (rng.next_float() * 2.0f - 1.0f) * scale;
            } else if (i < train::config::network_parameter_layout.global_rgb_offset) {
                const float scale = 1.0f / sqrtf(static_cast<float>(train::config::mlp_width));
                value             = (rng.next_float() * 2.0f - 1.0f) * scale;
            } else if (i == train::config::network_parameter_layout.global_rgb_offset) {
                value = 0.0f;
            } else {
                value = (rng.next_float() * 2.0f - 1.0f) * 1.0e-4f;
            }
            params_full_precision[i] = value;
            params[i]                = __float2half(value);
        }

        __global__ void cast_trainable_parameters_kernel(const float* __restrict__ params_full_precision, __half* __restrict__ params) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= train::config::network_parameter_layout.total_param_count) return;
            params[i] = __float2half(params_full_precision[i]);
        }

        __global__ void hash4_encode_forward_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, const __half* __restrict__ hash_params, __half* __restrict__ network_input) {
            const std::uint32_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t level        = blockIdx.y;
            if (sample_index >= sample_count || level >= train::config::hash4_level_count) return;

            const float* coord             = sample_coords + static_cast<std::uint64_t>(sample_index) * train::config::sample_coord_floats;
            const std::uint32_t resolution = hash4_level_resolution(level);
            const float scaled[4]          = {
                ::cuda::std::clamp(coord[0u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[1u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[2u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[3u], 0.0f, 1.0f) * static_cast<float>(resolution),
            };
            std::uint32_t base[4] = {};
            float frac[4]         = {};
            for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
                base[axis] = static_cast<std::uint32_t>(floorf(scaled[axis]));
                if (base[axis] >= resolution) {
                    base[axis] = resolution - 1u;
                    frac[axis] = 1.0f;
                } else {
                    frac[axis] = scaled[axis] - static_cast<float>(base[axis]);
                }
            }

            float features[2] = {};
            for (std::uint32_t corner = 0u; corner < 16u; ++corner) {
                const std::uint32_t xbit  = (corner >> 0u) & 1u;
                const std::uint32_t ybit  = (corner >> 1u) & 1u;
                const std::uint32_t zbit  = (corner >> 2u) & 1u;
                const std::uint32_t tbit  = (corner >> 3u) & 1u;
                const float weight        = (xbit != 0u ? frac[0u] : 1.0f - frac[0u]) * (ybit != 0u ? frac[1u] : 1.0f - frac[1u]) * (zbit != 0u ? frac[2u] : 1.0f - frac[2u]) * (tbit != 0u ? frac[3u] : 1.0f - frac[3u]);
                const std::uint32_t entry = hash4_entry_index(level, base[0u] + xbit, base[1u] + ybit, base[2u] + zbit, base[3u] + tbit);
                const __half* table       = hash_params + hash4_level_offset(level) + entry * train::config::hash4_features_per_level;
                features[0u] += weight * __half2float(table[0u]);
                features[1u] += weight * __half2float(table[1u]);
            }

            __half* output = network_input + static_cast<std::uint64_t>(sample_index) * train::config::hash4_output_width + level * train::config::hash4_features_per_level;
            output[0u]     = __float2half(features[0u]);
            output[1u]     = __float2half(features[1u]);
        }

        __global__ void hash4_encode_backward_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, const __half* __restrict__ network_input_gradients, float* __restrict__ param_gradients) {
            const std::uint32_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t level        = blockIdx.y;
            if (sample_index >= sample_count || level >= train::config::hash4_level_count) return;

            const float* coord             = sample_coords + static_cast<std::uint64_t>(sample_index) * train::config::sample_coord_floats;
            const std::uint32_t resolution = hash4_level_resolution(level);
            const float scaled[4]          = {
                ::cuda::std::clamp(coord[0u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[1u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[2u], 0.0f, 1.0f) * static_cast<float>(resolution),
                ::cuda::std::clamp(coord[3u], 0.0f, 1.0f) * static_cast<float>(resolution),
            };
            std::uint32_t base[4] = {};
            float frac[4]         = {};
            for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
                base[axis] = static_cast<std::uint32_t>(floorf(scaled[axis]));
                if (base[axis] >= resolution) {
                    base[axis] = resolution - 1u;
                    frac[axis] = 1.0f;
                } else {
                    frac[axis] = scaled[axis] - static_cast<float>(base[axis]);
                }
            }

            const __half* input_grad = network_input_gradients + static_cast<std::uint64_t>(sample_index) * train::config::hash4_output_width + level * train::config::hash4_features_per_level;
            for (std::uint32_t corner = 0u; corner < 16u; ++corner) {
                const std::uint32_t xbit  = (corner >> 0u) & 1u;
                const std::uint32_t ybit  = (corner >> 1u) & 1u;
                const std::uint32_t zbit  = (corner >> 2u) & 1u;
                const std::uint32_t tbit  = (corner >> 3u) & 1u;
                const float weight        = (xbit != 0u ? frac[0u] : 1.0f - frac[0u]) * (ybit != 0u ? frac[1u] : 1.0f - frac[1u]) * (zbit != 0u ? frac[2u] : 1.0f - frac[2u]) * (tbit != 0u ? frac[3u] : 1.0f - frac[3u]);
                const std::uint32_t entry = hash4_entry_index(level, base[0u] + xbit, base[1u] + ybit, base[2u] + zbit, base[3u] + tbit);
                float* table_grad         = param_gradients + train::config::network_parameter_layout.hash4_param_offset + hash4_level_offset(level) + entry * train::config::hash4_features_per_level;
                atomicAdd(table_grad + 0u, weight * __half2float(input_grad[0u]));
                atomicAdd(table_grad + 1u, weight * __half2float(input_grad[1u]));
            }
        }

        __global__ void relu_half_kernel(const std::uint32_t count, __half* __restrict__ values) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= count) return;
            const float value = __half2float(values[i]);
            values[i]         = __float2half(value > 0.0f ? value : 0.0f);
        }

        __global__ void prepare_output_gradient_kernel(const std::uint32_t count, const __half* __restrict__ output, __half* __restrict__ gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= count) return;
            const float output_value = __half2float(output[i]);
            const float gradient     = output_value > 0.0f ? __half2float(gradients[i]) : 0.0f;
            gradients[i]             = __float2half(gradient);
        }

        __global__ void prepare_hidden_gradient_kernel(const std::uint32_t count, const __half* __restrict__ hidden, __half* __restrict__ gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= count) return;
            const float hidden_value = __half2float(hidden[i]);
            const float gradient     = hidden_value > 0.0f ? __half2float(gradients[i]) : 0.0f;
            gradients[i]             = __float2half(gradient);
        }

        __device__ float3 read_training_rgb(const std::uint8_t* pixels, const std::uint32_t frame_index, const std::uint32_t pixel_x, const std::uint32_t pixel_y, const std::uint32_t width, const std::uint32_t height) {
            const std::uint64_t offset = (static_cast<std::uint64_t>(frame_index) * height * width + static_cast<std::uint64_t>(pixel_y) * width + pixel_x) * 4ull;
            constexpr float norm       = 1.0f / 255.0f;
            return {
                static_cast<float>(pixels[offset + 0u]) * norm,
                static_cast<float>(pixels[offset + 1u]) * norm,
                static_cast<float>(pixels[offset + 2u]) * norm,
            };
        }

        __device__ std::uint8_t to_rgb8(const float value) {
            const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
            return static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
        }

        __global__ void generate_evaluation_samples_kernel(const std::uint32_t tile_pixel_count, const std::uint32_t pixel_offset, const std::uint32_t sample_limit, const std::uint32_t render_width, const std::uint32_t view_index, const std::uint32_t time_index, const std::uint32_t time_count, const float* __restrict__ camera, const float* __restrict__ intrinsics, const std::uint32_t* __restrict__ frame_indices, const float* __restrict__ field_to_world_linear, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ overflow_counter, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= tile_pixel_count) return;

            numsteps_out[i * 2u + 0u] = 0u;
            numsteps_out[i * 2u + 1u] = 0u;

            const std::uint32_t frame_index = frame_indices[view_index * time_count + time_index];
            const float normalized_time     = time_count == 1u ? 0.0f : static_cast<float>(time_index) / static_cast<float>(time_count - 1u);
            const float* frame_camera       = camera + static_cast<std::uint64_t>(frame_index) * 12u;
            const float* frame_intrinsics   = intrinsics + static_cast<std::uint64_t>(frame_index) * 4u;

            const std::uint32_t pixel_index     = pixel_offset + i;
            const std::uint32_t pixel_x         = pixel_index % render_width;
            const std::uint32_t pixel_y         = pixel_index / render_width;
            const float sample_x                = static_cast<float>(pixel_x) + 0.5f;
            const float sample_y                = static_cast<float>(pixel_y) + 0.5f;
            const float ray_x                   = (sample_x - frame_intrinsics[2u]) / frame_intrinsics[0u];
            const float ray_y                   = (sample_y - frame_intrinsics[3u]) / frame_intrinsics[1u];
            const float3 camera_x               = {frame_camera[0u], frame_camera[1u], frame_camera[2u]};
            const float3 camera_y               = {frame_camera[3u], frame_camera[4u], frame_camera[5u]};
            const float3 camera_z               = {frame_camera[6u], frame_camera[7u], frame_camera[8u]};
            const float3 ray_origin             = {frame_camera[9u], frame_camera[10u], frame_camera[11u]};
            float3 ray_direction                = {
                camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
            };
            if (ray_direction.x == 0.0f && ray_direction.y == 0.0f && ray_direction.z == 0.0f) ray_direction = camera_z;

            const float direction_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
            if (direction_length == 0.0f) return;
            const float3 ray_direction_normalized = {ray_direction.x / direction_length, ray_direction.y / direction_length, ray_direction.z / direction_length};
            const float metric_per_field_unit     = field_metric_per_unit(field_to_world_linear, ray_direction_normalized);
            if (!isfinite(metric_per_field_unit) || metric_per_field_unit <= 0.0f) return;

            float tmin = 0.0f;
            float tmax = 0.0f;
            if (!intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin, tmax)) return;

            constexpr float dt    = train::config::evaluation_ray_stepsize;
            const float dt_metric = dt * metric_per_field_unit;
            const float start_t   = tmin + 0.5f * dt;

            std::uint32_t numsteps = 0u;
            for (float t = start_t; numsteps < train::config::evaluation_ray_steps && t <= tmax; t += dt) ++numsteps;

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base >= sample_limit) {
                atomicAdd(overflow_counter, 1u);
                return;
            }
            const std::uint32_t stored_numsteps = ::cuda::std::min(numsteps, sample_limit - base);
            if (stored_numsteps != numsteps) atomicAdd(overflow_counter, 1u);
            if (stored_numsteps == 0u) return;

            numsteps_out[i * 2u + 0u] = stored_numsteps;
            numsteps_out[i * 2u + 1u] = base;

            float t = start_t;
            for (std::uint32_t j = 0u; j < stored_numsteps; ++j) {
                const float3 pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                float* coord    = coords_out + static_cast<std::uint64_t>(base + j) * train::config::sample_coord_floats;
                coord[0u]       = pos.x;
                coord[1u]       = pos.y;
                coord[2u]       = pos.z;
                coord[3u]       = normalized_time;
                coord[4u]       = dt_metric;
                t += dt;
            }
        }

        __global__ void compose_evaluation_image_kernel(const std::uint32_t tile_pixel_count, const std::uint32_t pixel_offset, const std::uint32_t render_width, const std::uint32_t view_index, const std::uint32_t time_index, const std::uint32_t time_count, const std::uint32_t source_width, const std::uint32_t source_height, const std::uint8_t* __restrict__ pixels, const std::uint32_t* __restrict__ frame_indices, const __half* __restrict__ network_output, const std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, const __half* __restrict__ params, double* __restrict__ loss_sum, std::uint8_t* __restrict__ output_pixels) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= tile_pixel_count) return;

            const std::uint32_t frame_index = frame_indices[view_index * time_count + time_index];
            const std::uint32_t pixel_index = pixel_offset + i;
            const std::uint32_t pixel_x     = pixel_index % render_width;
            const std::uint32_t pixel_y     = pixel_index / render_width;
            const std::uint32_t numsteps    = numsteps_in[i * 2u + 0u];
            const std::uint32_t base        = numsteps_in[i * 2u + 1u];
            const __half* output            = network_output + static_cast<std::uint64_t>(base) * train::config::network_output_width;
            const float* coord              = coords_in + static_cast<std::uint64_t>(base) * train::config::sample_coord_floats;

            float transmittance = 1.0f;
            for (std::uint32_t j = 0u; j < numsteps; ++j) {
                if (transmittance < train::config::transmittance_epsilon) break;
                const float sigma = fmaxf(__half2float(output[0u]), 0.0f);
                const float alpha = 1.0f - __expf(-sigma * coord[4u]);
                transmittance *= 1.0f - alpha;
                output += train::config::network_output_width;
                coord += train::config::sample_coord_floats;
            }

            const float rgb_param    = __half2float(params[train::config::network_parameter_layout.global_rgb_offset]);
            const float color        = 0.6f + tanhf(rgb_param) * 0.4f;
            const float prediction   = color * (1.0f - transmittance);
            const float3 target      = read_training_rgb(pixels, frame_index, pixel_x, pixel_y, source_width, source_height);
            const float difference_x = prediction - target.x;
            const float difference_y = prediction - target.y;
            const float difference_z = prediction - target.z;
            atomicAdd(loss_sum, difference_x * difference_x + difference_y * difference_y + difference_z * difference_z);

            const std::uint64_t output_offset = static_cast<std::uint64_t>(pixel_index) * 3ull;
            const std::uint8_t rgb            = to_rgb8(prediction);
            output_pixels[output_offset + 0u] = rgb;
            output_pixels[output_offset + 1u] = rgb;
            output_pixels[output_offset + 2u] = rgb;
        }

        __global__ void compute_training_loss_and_compact_kernel(const std::uint32_t rays_per_batch, const std::uint32_t current_step, const std::uint32_t* __restrict__ ray_counter, const std::uint8_t* __restrict__ pixels, const std::uint32_t* __restrict__ frame_indices, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, const __half* __restrict__ network_output, std::uint32_t* __restrict__ compacted_sample_counter, const std::uint32_t* __restrict__ ray_indices_in, std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, float* __restrict__ coords_out, __half* __restrict__ dloss_doutput, float* __restrict__ param_gradients, const __half* __restrict__ params, float* __restrict__ loss_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= *ray_counter) return;

            const std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
            const std::uint32_t base     = numsteps_in[i * 2u + 1u];
            const __half* output         = network_output + static_cast<std::uint64_t>(base) * train::config::network_output_width;
            const float* coord_in        = coords_in + static_cast<std::uint64_t>(base) * train::config::sample_coord_floats;

            float transmittance              = 1.0f;
            float final_transmittance        = 1.0f;
            std::uint32_t compacted_numsteps = 0u;
            for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
                if (transmittance < train::config::transmittance_epsilon) break;
                const float sigma = fmaxf(__half2float(output[0u]), 0.0f);
                const float dt    = coord_in[4u];
                const float alpha = 1.0f - __expf(-sigma * dt);
                transmittance *= 1.0f - alpha;
                final_transmittance = transmittance;
                output += train::config::network_output_width;
                coord_in += train::config::sample_coord_floats;
            }

            const std::uint32_t ray_index  = ray_indices_in[i];
            std::uint32_t view_index       = 0u;
            std::uint32_t floor_time_index = 0u;
            std::uint32_t ceil_time_index  = 0u;
            float time_residual            = 0.0f;
            std::uint32_t pixel_x          = 0u;
            std::uint32_t pixel_y          = 0u;
            float ignored_ray_phase        = 0.0f;
            replay_training_ray_rng(ray_index, current_step, view_count, time_count, width, height, view_index, floor_time_index, ceil_time_index, time_residual, pixel_x, pixel_y, ignored_ray_phase);
            const std::uint32_t floor_frame_index = frame_indices[view_index * time_count + floor_time_index];
            const std::uint32_t ceil_frame_index  = frame_indices[view_index * time_count + ceil_time_index];
            const float3 floor_rgb                = read_training_rgb(pixels, floor_frame_index, pixel_x, pixel_y, width, height);
            const float3 ceil_rgb                 = read_training_rgb(pixels, ceil_frame_index, pixel_x, pixel_y, width, height);
            const float3 target                   = {
                floor_rgb.x * (1.0f - time_residual) + ceil_rgb.x * time_residual,
                floor_rgb.y * (1.0f - time_residual) + ceil_rgb.y * time_residual,
                floor_rgb.z * (1.0f - time_residual) + ceil_rgb.z * time_residual,
            };

            const float rgb_param   = __half2float(params[train::config::network_parameter_layout.global_rgb_offset]);
            const float tanh_rgb    = tanhf(rgb_param);
            const float color       = 0.6f + tanh_rgb * 0.4f;
            const float opacity     = 1.0f - final_transmittance;
            const float3 prediction = {color * opacity, color * opacity, color * opacity};
            const float3 difference = {prediction.x - target.x, prediction.y - target.y, prediction.z - target.z};
            if (loss_output != nullptr) loss_output[i] = (difference.x * difference.x + difference.y * difference.y + difference.z * difference.z) / (3.0f * static_cast<float>(rays_per_batch));

            const float dloss_dprediction_x = 2.0f * difference.x / (3.0f * static_cast<float>(rays_per_batch));
            const float dloss_dprediction_y = 2.0f * difference.y / (3.0f * static_cast<float>(rays_per_batch));
            const float dloss_dprediction_z = 2.0f * difference.z / (3.0f * static_cast<float>(rays_per_batch));
            const float dloss_dopacity      = color * (dloss_dprediction_x + dloss_dprediction_y + dloss_dprediction_z);
            const float dloss_dcolor        = opacity * (dloss_dprediction_x + dloss_dprediction_y + dloss_dprediction_z);
            atomicAdd(param_gradients + train::config::network_parameter_layout.global_rgb_offset, train::config::optimizer_loss_scale * dloss_dcolor * 0.4f * (1.0f - tanh_rgb * tanh_rgb));

            output                              = network_output + static_cast<std::uint64_t>(base) * train::config::network_output_width;
            coord_in                            = coords_in + static_cast<std::uint64_t>(base) * train::config::sample_coord_floats;
            const std::uint32_t compacted_base  = atomicAdd(compacted_sample_counter, compacted_numsteps);
            const std::uint32_t remaining_slots = compacted_base < train::config::network_batch_size ? train::config::network_batch_size - compacted_base : 0u;
            compacted_numsteps                  = compacted_numsteps < remaining_slots ? compacted_numsteps : remaining_slots;
            numsteps_in[i * 2u + 0u]            = compacted_numsteps;
            numsteps_in[i * 2u + 1u]            = compacted_base;
            if (compacted_numsteps == 0u) return;

            coords_out += static_cast<std::uint64_t>(compacted_base) * train::config::sample_coord_floats;
            dloss_doutput += static_cast<std::uint64_t>(compacted_base) * train::config::network_output_width;

            transmittance = 1.0f;
            for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
                const float sigma                = fmaxf(__half2float(output[0u]), 0.0f);
                const float dt                   = coord_in[4u];
                const float alpha                = 1.0f - __expf(-sigma * dt);
                const float one_minus_alpha      = fmaxf(1.0f - alpha, 1.0e-12f);
                const float suffix_transmittance = final_transmittance / fmaxf(transmittance * one_minus_alpha, 1.0e-12f);
                const float dalpha_dsigma        = dt * __expf(-sigma * dt);
                const float dopacity_dsigma      = transmittance * suffix_transmittance * dalpha_dsigma;
                dloss_doutput[j]                 = __float2half(train::config::optimizer_loss_scale * dloss_dopacity * dopacity_dsigma);

                float* coord_out   = coords_out + static_cast<std::uint64_t>(j) * train::config::sample_coord_floats;
                const float* coord = coord_in + static_cast<std::uint64_t>(j) * train::config::sample_coord_floats;
                for (std::uint32_t k = 0u; k < train::config::sample_coord_floats; ++k) coord_out[k] = coord[k];

                transmittance *= one_minus_alpha;
                output += train::config::network_output_width;
                coord_in += train::config::sample_coord_floats;
            }
        }

        __global__ void pad_rollover_coords_kernel(const std::uint32_t* __restrict__ input_count, float* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * train::config::sample_coord_floats || i >= train::config::network_batch_size * train::config::sample_coord_floats || n == 0u) return;
            inout[i] = inout[i % (n * train::config::sample_coord_floats)];
        }

        __global__ void pad_rollover_network_output_gradients_kernel(const std::uint32_t* __restrict__ input_count, __half* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * train::config::network_output_width || i >= train::config::network_batch_size * train::config::network_output_width || n == 0u) return;
            inout[i] = __float2half(__half2float(inout[i % (n * train::config::network_output_width)]) * static_cast<float>(n) / static_cast<float>(train::config::network_batch_size));
        }

        __global__ void radam_step_kernel(float* __restrict__ params_full_precision, __half* __restrict__ params, const float* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= train::config::network_parameter_layout.total_param_count) return;

            float gradient         = gradients[i] / train::config::optimizer_loss_scale;
            const bool model_param = i < train::config::network_parameter_layout.mlp_param_count;
            if (model_param) gradient += train::config::optimizer_weight_decay * params_full_precision[i];
            const float gradient_sq  = gradient * gradient;
            const float first_moment = first_moments[i] = train::config::optimizer_beta1 * first_moments[i] + (1.0f - train::config::optimizer_beta1) * gradient;
            const float second_moment = second_moments[i] = train::config::optimizer_beta2 * second_moments[i] + (1.0f - train::config::optimizer_beta2) * gradient_sq;
            const std::uint32_t step                      = ++param_steps[i];
            const float beta2_t                           = powf(train::config::optimizer_beta2, static_cast<float>(step));
            constexpr float n_sma_max                     = 2.0f / (1.0f - train::config::optimizer_beta2) - 1.0f;
            const float n_sma                             = n_sma_max - 2.0f * static_cast<float>(step) * beta2_t / (1.0f - beta2_t);
            if (n_sma < 5.0f) return;

            const float step_size     = sqrtf((1.0f - beta2_t) * (n_sma - 4.0f) / (n_sma_max - 4.0f) * (n_sma - 2.0f) / n_sma * n_sma_max / (n_sma_max - 2.0f)) / (1.0f - powf(train::config::optimizer_beta1, static_cast<float>(step)));
            const float updated_param = params_full_precision[i] - train::config::optimizer_learning_rate * step_size * first_moment / (sqrtf(second_moment) + train::config::optimizer_epsilon);
            params_full_precision[i]  = updated_param;
            params[i]                 = __float2half(updated_param);
        }
    } // namespace

    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] != nullptr) cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    void upload_dataset(const std::uint8_t* const pixels, const std::size_t pixel_bytes, const float* const camera, const std::size_t camera_count, const float* const intrinsics, const std::size_t intrinsics_count, const std::uint32_t* const frame_indices, const std::size_t frame_index_count, const std::uint8_t*& out_pixels, const float*& out_camera, const float*& out_intrinsics, const std::uint32_t*& out_frame_indices) {
        out_pixels        = nullptr;
        out_camera        = nullptr;
        out_intrinsics    = nullptr;
        out_frame_indices = nullptr;

        if (pixels == nullptr || pixel_bytes == 0u || camera == nullptr || camera_count == 0u || intrinsics == nullptr || intrinsics_count == 0u || frame_indices == nullptr || frame_index_count == 0u) throw std::runtime_error{"invalid dynamic dataset upload input."};
        const std::size_t camera_bytes      = camera_count * sizeof(float);
        const std::size_t intrinsics_bytes  = intrinsics_count * sizeof(float);
        const std::size_t frame_index_bytes = frame_index_count * sizeof(std::uint32_t);

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

        void* uploaded_frame_indices = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_frame_indices, frame_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc frame indices failed: "} + cudaGetErrorString(status)};
        out_frame_indices = static_cast<std::uint32_t*>(uploaded_frame_indices);
        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint32_t*>(out_frame_indices), frame_indices, frame_index_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy frame indices failed: "} + cudaGetErrorString(status)};
    }

    void upload_field_domain(const float* const field_to_world_linear, float*& out_field_to_world_linear) {
        out_field_to_world_linear = nullptr;
        if (field_to_world_linear == nullptr) throw std::runtime_error{"invalid field domain upload input."};
        if (const cudaError_t status = cudaMalloc(&out_field_to_world_linear, 9u * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc field_to_world_linear failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(out_field_to_world_linear, field_to_world_linear, 9u * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy field_to_world_linear failed: "} + cudaGetErrorString(status)};
    }

    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy, std::uint32_t*& out_occupancy_grid_occupied_count) {
        out_sample_coords                 = nullptr;
        out_rays                          = nullptr;
        out_ray_indices                   = nullptr;
        out_numsteps                      = nullptr;
        out_ray_counter                   = nullptr;
        out_sample_counter                = nullptr;
        out_occupancy                     = nullptr;
        out_occupancy_grid_occupied_count = nullptr;

        constexpr std::size_t sample_coord_bytes = static_cast<std::size_t>(train::config::max_samples) * train::config::sample_coord_floats * sizeof(float);
        constexpr std::size_t ray_bytes          = static_cast<std::size_t>(train::config::initial_rays_per_batch) * train::config::ray_floats * sizeof(float);
        constexpr std::size_t ray_index_bytes    = static_cast<std::size_t>(train::config::initial_rays_per_batch) * sizeof(std::uint32_t);
        constexpr std::size_t numstep_bytes      = static_cast<std::size_t>(train::config::initial_rays_per_batch) * 2u * sizeof(std::uint32_t);
        constexpr std::size_t occupancy_bytes    = train::config::nerf_grid_cells / 8u;

        if (const cudaError_t status = cudaMalloc(&out_sample_coords, sample_coord_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample coords failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rays, ray_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler rays failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_indices, ray_index_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray indices failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_numsteps, numstep_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler numsteps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_occupancy, occupancy_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_occupancy_grid_occupied_count, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy grid occupied count failed: "} + cudaGetErrorString(status)};
    }

    void set_occupancy_grid_full(std::uint8_t* const occupancy, std::uint32_t* const occupancy_grid_occupied_count) {
        if (occupancy == nullptr || occupancy_grid_occupied_count == nullptr) throw std::runtime_error{"invalid occupancy grid full update input."};
        constexpr std::size_t occupancy_bytes = train::config::nerf_grid_cells / 8u;
        if (const cudaError_t status = cudaMemset(occupancy, 0xFF, occupancy_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset full occupancy failed: "} + cudaGetErrorString(status)};
        constexpr std::uint32_t occupied_count = train::config::nerf_grid_cells;
        if (const cudaError_t status = cudaMemcpy(occupancy_grid_occupied_count, &occupied_count, sizeof(std::uint32_t), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy full occupancy count failed: "} + cudaGetErrorString(status)};
    }

    void sample_training_batch(const float* const camera, const float* const intrinsics, const std::uint32_t* const frame_indices, const float* const field_to_world_linear, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, const std::uint32_t current_step, const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint8_t* const occupancy, float* const sample_coords, float* const rays, std::uint32_t* const ray_indices, std::uint32_t* const numsteps, std::uint32_t* const ray_counter, std::uint32_t* const sample_counter) {
        if (camera == nullptr || intrinsics == nullptr || frame_indices == nullptr || field_to_world_linear == nullptr || occupancy == nullptr || sample_coords == nullptr || rays == nullptr || ray_indices == nullptr || numsteps == nullptr || ray_counter == nullptr || sample_counter == nullptr) throw std::runtime_error{"invalid training sampler input."};
        if (view_count == 0u || time_count == 0u || view_count > std::numeric_limits<std::uint32_t>::max() / time_count || width == 0u || height == 0u || rays_per_batch == 0u || rays_per_batch > train::config::initial_rays_per_batch || sample_limit == 0u || sample_limit > train::config::max_samples) throw std::runtime_error{"invalid training sampler shape."};
        if (const cudaError_t status = cudaMemset(ray_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler sample counter failed: "} + cudaGetErrorString(status)};
        const std::uint32_t blocks = (rays_per_batch + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        generate_training_samples_kernel<<<blocks, train::config::threads_per_block>>>(rays_per_batch, sample_limit, current_step, view_count, time_count, width, height, camera, intrinsics, frame_indices, field_to_world_linear, occupancy, ray_counter, sample_counter, ray_indices, rays, numsteps, sample_coords);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate_training_samples_kernel failed: "} + cudaGetErrorString(status)};
    }

    void allocate_training_loss_buffers(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values, std::uint16_t*& out_network_output_gradients) {
        out_compacted_sample_counter = nullptr;
        out_compacted_sample_coords  = nullptr;
        out_loss_values              = nullptr;
        out_network_output_gradients = nullptr;
        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc compacted sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_coords, static_cast<std::size_t>(train::config::network_batch_size) * train::config::sample_coord_floats * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc compacted sample coords failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_loss_values, static_cast<std::size_t>(train::config::initial_rays_per_batch) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc loss values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_output_gradients, static_cast<std::size_t>(train::config::network_batch_size) * train::config::network_output_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network output gradients failed: "} + cudaGetErrorString(status)};
    }

    void allocate_evaluation_buffers(const std::uint32_t render_pixel_capacity, std::uint32_t*& out_evaluation_numsteps, std::uint32_t*& out_evaluation_sample_counter, std::uint32_t*& out_evaluation_overflow_counter, double*& out_evaluation_loss_sum, std::uint8_t*& out_evaluation_pixels) {
        out_evaluation_numsteps         = nullptr;
        out_evaluation_sample_counter   = nullptr;
        out_evaluation_overflow_counter = nullptr;
        out_evaluation_loss_sum         = nullptr;
        out_evaluation_pixels           = nullptr;
        if (render_pixel_capacity == 0u) throw std::runtime_error{"evaluation render pixel capacity must be positive."};
        if (const cudaError_t status = cudaMalloc(&out_evaluation_numsteps, static_cast<std::size_t>(train::config::evaluation_tile_pixels) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc evaluation numsteps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_evaluation_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc evaluation sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_evaluation_overflow_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc evaluation overflow counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_evaluation_loss_sum, sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc evaluation loss sum failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_evaluation_pixels, static_cast<std::size_t>(render_pixel_capacity) * 3u * sizeof(std::uint8_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc evaluation pixels failed: "} + cudaGetErrorString(status)};
    }

    void allocate_network_buffers(std::uint16_t*& out_network_input, std::uint16_t*& out_network_hidden, std::uint16_t*& out_network_output, std::uint16_t*& out_network_input_gradients, std::uint16_t*& out_network_hidden_gradients, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace) {
        out_network_input            = nullptr;
        out_network_hidden           = nullptr;
        out_network_output           = nullptr;
        out_network_input_gradients  = nullptr;
        out_network_hidden_gradients = nullptr;
        out_cublaslt_handle          = nullptr;
        out_cublaslt_workspace       = nullptr;

        cublasLtHandle_t handle = nullptr;
        if (const cublasStatus_t status = cublasLtCreate(&handle); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtCreate failed: "} + cublasGetStatusString(status)};
        out_cublaslt_handle = reinterpret_cast<void*>(handle);
        if (const cudaError_t status = cudaMalloc(&out_network_input, static_cast<std::size_t>(train::config::network_batch_size) * train::config::hash4_output_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network input failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_hidden, static_cast<std::size_t>(train::config::network_batch_size) * train::config::mlp_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network hidden failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_output, static_cast<std::size_t>(train::config::max_samples) * train::config::network_output_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network output failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_input_gradients, static_cast<std::size_t>(train::config::network_batch_size) * train::config::hash4_output_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network input gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_hidden_gradients, static_cast<std::size_t>(train::config::network_batch_size) * train::config::mlp_width * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network hidden gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_cublaslt_workspace, train::config::cublaslt_workspace_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc cublasLt workspace failed: "} + cudaGetErrorString(status)};
    }

    void destroy_network_handle(void*& cublaslt_handle) noexcept {
        if (cublaslt_handle != nullptr) {
            cublasLtDestroy(static_cast<cublasLtHandle_t>(cublaslt_handle));
            cublaslt_handle = nullptr;
        }
    }

    void allocate_trainable_parameter_buffers(float*& out_params_full_precision, std::uint16_t*& out_params, float*& out_param_gradients) {
        out_params_full_precision = nullptr;
        out_params                = nullptr;
        out_param_gradients       = nullptr;
        if (const cudaError_t status = cudaMalloc(&out_params_full_precision, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable params full precision failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_params, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable params failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_param_gradients, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable gradients failed: "} + cudaGetErrorString(status)};
    }

    void initialize_trainable_parameters(float* const params_full_precision, std::uint16_t* const params) {
        if (params_full_precision == nullptr || params == nullptr) throw std::runtime_error{"invalid trainable parameter initialization input."};
        constexpr std::uint32_t blocks = (train::config::network_parameter_layout.total_param_count + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        initialize_trainable_parameters_kernel<<<blocks, train::config::threads_per_block>>>(params_full_precision, reinterpret_cast<__half*>(params));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"initialize_trainable_parameters_kernel failed: "} + cudaGetErrorString(status)};
    }

    void download_trainable_parameters(const float* const params_full_precision, float* const out_params_full_precision) {
        if (params_full_precision == nullptr || out_params_full_precision == nullptr) throw std::runtime_error{"invalid trainable parameter download input."};
        if (const cudaError_t status = cudaMemcpy(out_params_full_precision, params_full_precision, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy trainable params download failed: "} + cudaGetErrorString(status)};
    }

    void upload_trainable_parameters(const float* const params_full_precision, float* const out_params_full_precision, std::uint16_t* const out_params, float* const out_param_gradients, float* const optimizer_first_moments, float* const optimizer_second_moments, std::uint32_t* const optimizer_param_steps) {
        if (params_full_precision == nullptr || out_params_full_precision == nullptr || out_params == nullptr || out_param_gradients == nullptr || optimizer_first_moments == nullptr || optimizer_second_moments == nullptr || optimizer_param_steps == nullptr) throw std::runtime_error{"invalid trainable parameter upload input."};
        constexpr std::size_t param_bytes = static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float);
        if (const cudaError_t status = cudaMemcpy(out_params_full_precision, params_full_precision, param_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy trainable params upload failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_param_gradients, 0, param_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset uploaded trainable gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_first_moments, 0, param_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset uploaded optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_second_moments, 0, param_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset uploaded optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_param_steps, 0, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset uploaded optimizer param steps failed: "} + cudaGetErrorString(status)};
        constexpr std::uint32_t blocks = (train::config::network_parameter_layout.total_param_count + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        cast_trainable_parameters_kernel<<<blocks, train::config::threads_per_block>>>(out_params_full_precision, reinterpret_cast<__half*>(out_params));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"cast_trainable_parameters_kernel failed: "} + cudaGetErrorString(status)};
    }

    void allocate_optimizer_buffers(float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps) {
        out_first_moments  = nullptr;
        out_second_moments = nullptr;
        out_param_steps    = nullptr;
        if (const cudaError_t status = cudaMalloc(&out_first_moments, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_second_moments, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_param_steps, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer param steps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_first_moments, 0, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_second_moments, 0, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_param_steps, 0, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer param steps failed: "} + cudaGetErrorString(status)};
    }

    void evaluate_network(const std::uint32_t sample_count, const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const network_input, std::uint16_t* const network_hidden, std::uint16_t* const network_output, void* const cublaslt_handle, std::uint8_t* const cublaslt_workspace) {
        if (sample_count == 0u) return;
        if (sample_count > train::config::network_batch_size || sample_coords == nullptr || params == nullptr || network_input == nullptr || network_hidden == nullptr || network_output == nullptr || cublaslt_handle == nullptr || cublaslt_workspace == nullptr) throw std::runtime_error{"invalid network evaluation input."};
        constexpr dim3 grid_blocks{(train::config::network_batch_size + train::config::grid_forward_threads - 1u) / train::config::grid_forward_threads, train::config::hash4_level_count, 1u};
        hash4_encode_forward_kernel<<<grid_blocks, train::config::grid_forward_threads>>>(sample_count, sample_coords, reinterpret_cast<const __half*>(params + train::config::network_parameter_layout.hash4_param_offset), reinterpret_cast<__half*>(network_input));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"hash4_encode_forward_kernel failed: "} + cudaGetErrorString(status)};
        const auto cublaslt = static_cast<cublasLtHandle_t>(cublaslt_handle);
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, network_input, static_cast<int>(sample_count), train::config::mlp_input_width, params + train::config::network_parameter_layout.mlp_input_weight_offset, train::config::mlp_width, train::config::mlp_input_width, network_hidden, static_cast<int>(sample_count), train::config::mlp_width, 1.0f, 0.0f, cublaslt_workspace);
        relu_half_kernel<<<(sample_count * train::config::mlp_width + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(sample_count * train::config::mlp_width, reinterpret_cast<__half*>(network_hidden));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"relu hidden failed: "} + cudaGetErrorString(status)};
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, network_hidden, static_cast<int>(sample_count), train::config::mlp_width, params + train::config::network_parameter_layout.mlp_output_weight_offset, train::config::network_output_width, train::config::mlp_width, network_output, static_cast<int>(sample_count), train::config::network_output_width, 1.0f, 0.0f, cublaslt_workspace);
        relu_half_kernel<<<(sample_count + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(sample_count, reinterpret_cast<__half*>(network_output));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"relu output failed: "} + cudaGetErrorString(status)};
    }

    void run_evaluation_image(const std::uint8_t* const pixels, const float* const camera, const float* const intrinsics, const std::uint32_t* const frame_indices, const float* const field_to_world_linear, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, const std::uint32_t view_index, const std::uint32_t time_index, const std::uint32_t render_pixel_capacity, const std::uint16_t* const params, float* const sample_coords, std::uint16_t* const network_input, std::uint16_t* const network_hidden, std::uint16_t* const network_output, void* const cublaslt_handle, std::uint8_t* const cublaslt_workspace, std::uint32_t* const evaluation_numsteps, std::uint32_t* const evaluation_sample_counter, std::uint32_t* const evaluation_overflow_counter, double* const evaluation_loss_sum, std::uint8_t* const evaluation_pixels, std::uint8_t* const host_evaluation_pixels, double& out_loss_sum) {
        out_loss_sum = 0.0;
        if (pixels == nullptr || camera == nullptr || intrinsics == nullptr || frame_indices == nullptr || field_to_world_linear == nullptr || params == nullptr || sample_coords == nullptr || network_input == nullptr || network_hidden == nullptr || network_output == nullptr || cublaslt_handle == nullptr || cublaslt_workspace == nullptr || evaluation_numsteps == nullptr || evaluation_sample_counter == nullptr || evaluation_overflow_counter == nullptr || evaluation_loss_sum == nullptr || evaluation_pixels == nullptr || host_evaluation_pixels == nullptr) throw std::runtime_error{"invalid evaluation image input."};
        if (view_count == 0u || time_count == 0u || view_index >= view_count || time_index >= time_count || width == 0u || height == 0u) throw std::runtime_error{"invalid evaluation image shape."};
        const std::uint32_t render_width     = width;
        const std::uint32_t render_height    = height;
        const std::uint64_t render_pixels_64 = static_cast<std::uint64_t>(render_width) * render_height;
        if (render_pixels_64 == 0ull || render_pixels_64 > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"evaluation image has invalid render dimensions."};
        const auto render_pixels = static_cast<std::uint32_t>(render_pixels_64);
        if (render_pixels > render_pixel_capacity) throw std::runtime_error{"evaluation image exceeds allocated render pixel capacity."};

        if (const cudaError_t status = cudaMemset(evaluation_loss_sum, 0, sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset evaluation loss sum failed: "} + cudaGetErrorString(status)};
        for (std::uint32_t pixel_offset = 0u; pixel_offset < render_pixels; pixel_offset += train::config::evaluation_tile_pixels) {
            const std::uint32_t tile_pixels = ::cuda::std::min(train::config::evaluation_tile_pixels, render_pixels - pixel_offset);
            if (const cudaError_t status = cudaMemset(evaluation_numsteps, 0, static_cast<std::size_t>(tile_pixels) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset evaluation numsteps failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemset(evaluation_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset evaluation sample counter failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemset(evaluation_overflow_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset evaluation overflow counter failed: "} + cudaGetErrorString(status)};

            generate_evaluation_samples_kernel<<<(tile_pixels + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(tile_pixels, pixel_offset, train::config::max_samples, render_width, view_index, time_index, time_count, camera, intrinsics, frame_indices, field_to_world_linear, evaluation_sample_counter, evaluation_overflow_counter, evaluation_numsteps, sample_coords);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate_evaluation_samples_kernel failed: "} + cudaGetErrorString(status)};

            std::uint32_t overflow_count = 0u;
            read_counter(evaluation_overflow_counter, overflow_count);
            if (overflow_count != 0u) throw std::runtime_error{"evaluation sample generation exceeded network batch capacity."};
            std::uint32_t sample_count = 0u;
            read_counter(evaluation_sample_counter, sample_count);
            if (sample_count > train::config::network_batch_size) throw std::runtime_error{"evaluation sample count exceeds network batch size."};
            evaluate_network(sample_count, sample_coords, params, network_input, network_hidden, network_output, cublaslt_handle, cublaslt_workspace);

            compose_evaluation_image_kernel<<<(tile_pixels + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(tile_pixels, pixel_offset, render_width, view_index, time_index, time_count, width, height, pixels, frame_indices, reinterpret_cast<const __half*>(network_output), evaluation_numsteps, sample_coords, reinterpret_cast<const __half*>(params), evaluation_loss_sum, evaluation_pixels);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compose_evaluation_image_kernel failed: "} + cudaGetErrorString(status)};
        }

        if (const cudaError_t status = cudaMemcpy(&out_loss_sum, evaluation_loss_sum, sizeof(double), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy evaluation loss sum failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(host_evaluation_pixels, evaluation_pixels, static_cast<std::size_t>(render_pixels) * 3u * sizeof(std::uint8_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy evaluation pixels failed: "} + cudaGetErrorString(status)};
    }

    void forward_network(const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const network_input, std::uint16_t* const network_hidden, std::uint16_t* const network_output, void* const cublaslt_handle, std::uint8_t* const cublaslt_workspace) {
        evaluate_network(train::config::network_batch_size, sample_coords, params, network_input, network_hidden, network_output, cublaslt_handle, cublaslt_workspace);
    }

    void compute_training_loss_and_compact_samples(const std::uint32_t rays_per_batch, const std::uint32_t current_step, const std::uint32_t* const ray_counter, const std::uint8_t* const pixels, const std::uint32_t* const frame_indices, const std::uint32_t view_count, const std::uint32_t time_count, const std::uint32_t width, const std::uint32_t height, const std::uint16_t* const network_output, std::uint32_t* const compacted_sample_counter, const std::uint32_t* const ray_indices, std::uint32_t* const numsteps, const float* const sample_coords, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients, float* const param_gradients, const std::uint16_t* const params, float* const loss_values) {
        if (rays_per_batch == 0u) return;
        if (ray_counter == nullptr || pixels == nullptr || frame_indices == nullptr || view_count == 0u || time_count == 0u || width == 0u || height == 0u || network_output == nullptr || compacted_sample_counter == nullptr || ray_indices == nullptr || numsteps == nullptr || sample_coords == nullptr || compacted_sample_coords == nullptr || network_output_gradients == nullptr || param_gradients == nullptr || params == nullptr || loss_values == nullptr) throw std::runtime_error{"invalid loss and compaction input."};
        if (const cudaError_t status = cudaMemset(compacted_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset compacted sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(loss_values, 0, static_cast<std::size_t>(rays_per_batch) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loss values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(param_gradients, 0, static_cast<std::size_t>(train::config::network_parameter_layout.total_param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset trainable gradients failed: "} + cudaGetErrorString(status)};
        const std::uint32_t blocks = (rays_per_batch + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        compute_training_loss_and_compact_kernel<<<blocks, train::config::threads_per_block>>>(rays_per_batch, current_step, ray_counter, pixels, frame_indices, view_count, time_count, width, height, reinterpret_cast<const __half*>(network_output), compacted_sample_counter, ray_indices, numsteps, sample_coords, compacted_sample_coords, reinterpret_cast<__half*>(network_output_gradients), param_gradients, reinterpret_cast<const __half*>(params), loss_values);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_training_loss_and_compact_kernel failed: "} + cudaGetErrorString(status)};
    }

    void pad_compacted_training_batch(const std::uint32_t* const compacted_sample_counter, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients) {
        if (compacted_sample_counter == nullptr || compacted_sample_coords == nullptr || network_output_gradients == nullptr) throw std::runtime_error{"rollover buffers are null."};
        constexpr std::uint32_t gradient_elements = train::config::network_batch_size * train::config::network_output_width;
        pad_rollover_network_output_gradients_kernel<<<(gradient_elements + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(compacted_sample_counter, reinterpret_cast<__half*>(network_output_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"pad_rollover_network_output_gradients_kernel failed: "} + cudaGetErrorString(status)};
        constexpr std::uint32_t coord_elements = train::config::network_batch_size * train::config::sample_coord_floats;
        pad_rollover_coords_kernel<<<(coord_elements + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(compacted_sample_counter, compacted_sample_coords);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"pad_rollover_coords_kernel failed: "} + cudaGetErrorString(status)};
    }

    void backward_network(const float* const sample_coords, const std::uint16_t* const params, float* const param_gradients, const std::uint16_t* const network_input, const std::uint16_t* const network_hidden, const std::uint16_t* const network_output, const std::uint16_t* const network_output_gradients, std::uint16_t* const network_input_gradients, std::uint16_t* const network_hidden_gradients, void* const cublaslt_handle, std::uint8_t* const cublaslt_workspace) {
        if (sample_coords == nullptr || params == nullptr || param_gradients == nullptr || network_input == nullptr || network_hidden == nullptr || network_output == nullptr || network_output_gradients == nullptr || network_input_gradients == nullptr || network_hidden_gradients == nullptr || cublaslt_handle == nullptr || cublaslt_workspace == nullptr) throw std::runtime_error{"invalid network backward input."};
        prepare_output_gradient_kernel<<<(train::config::network_batch_size + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(train::config::network_batch_size, reinterpret_cast<const __half*>(network_output), reinterpret_cast<__half*>(const_cast<std::uint16_t*>(network_output_gradients)));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"prepare_output_gradient_kernel failed: "} + cudaGetErrorString(status)};
        const auto cublaslt = static_cast<cublasLtHandle_t>(cublaslt_handle);
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, network_output_gradients, train::config::network_batch_size, train::config::network_output_width, network_hidden, train::config::network_batch_size, train::config::mlp_width, param_gradients + train::config::network_parameter_layout.mlp_output_weight_offset, train::config::network_output_width, train::config::mlp_width, 1.0f, 0.0f, cublaslt_workspace);
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, network_output_gradients, train::config::network_batch_size, train::config::network_output_width, params + train::config::network_parameter_layout.mlp_output_weight_offset, train::config::network_output_width, train::config::mlp_width, network_hidden_gradients, train::config::network_batch_size, train::config::mlp_width, 1.0f, 0.0f, cublaslt_workspace);
        prepare_hidden_gradient_kernel<<<(train::config::network_batch_size * train::config::mlp_width + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(train::config::network_batch_size * train::config::mlp_width, reinterpret_cast<const __half*>(network_hidden), reinterpret_cast<__half*>(network_hidden_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"prepare_hidden_gradient_kernel failed: "} + cudaGetErrorString(status)};
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, network_hidden_gradients, train::config::network_batch_size, train::config::mlp_width, network_input, train::config::network_batch_size, train::config::mlp_input_width, param_gradients + train::config::network_parameter_layout.mlp_input_weight_offset, train::config::mlp_width, train::config::mlp_input_width, 1.0f, 0.0f, cublaslt_workspace);
        cublaslt_matmul_row_major(cublaslt, CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, network_hidden_gradients, train::config::network_batch_size, train::config::mlp_width, params + train::config::network_parameter_layout.mlp_input_weight_offset, train::config::mlp_width, train::config::mlp_input_width, network_input_gradients, train::config::network_batch_size, train::config::mlp_input_width, 1.0f, 0.0f, cublaslt_workspace);
        constexpr dim3 grid_blocks{(train::config::network_batch_size + train::config::grid_backward_threads - 1u) / train::config::grid_backward_threads, train::config::hash4_level_count, 1u};
        hash4_encode_backward_kernel<<<grid_blocks, train::config::grid_backward_threads>>>(train::config::network_batch_size, sample_coords, reinterpret_cast<const __half*>(network_input_gradients), param_gradients);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"hash4_encode_backward_kernel failed: "} + cudaGetErrorString(status)};
    }

    void step_optimizer(float* const params_full_precision, std::uint16_t* const params, const float* const gradients, float* const first_moments, float* const second_moments, std::uint32_t* const param_steps) {
        if (params_full_precision == nullptr || params == nullptr || gradients == nullptr || first_moments == nullptr || second_moments == nullptr || param_steps == nullptr) throw std::runtime_error{"invalid optimizer input."};
        constexpr std::uint32_t blocks = (train::config::network_parameter_layout.total_param_count + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        radam_step_kernel<<<blocks, train::config::threads_per_block>>>(params_full_precision, reinterpret_cast<__half*>(params), gradients, first_moments, second_moments, param_steps);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"radam_step_kernel failed: "} + cudaGetErrorString(status)};
    }

    void read_counter(const std::uint32_t* const counter, std::uint32_t& value) {
        if (counter == nullptr) throw std::runtime_error{"counter read input is null."};
        if (const cudaError_t status = cudaMemcpy(&value, counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy counter read failed: "} + cudaGetErrorString(status)};
    }

    void read_loss_sum(const float* const loss_values, const std::uint32_t loss_count, float& out_loss_sum) {
        out_loss_sum = 0.0f;
        if (loss_count == 0u) return;
        if (loss_values == nullptr) throw std::runtime_error{"loss values are null."};
        std::vector<float> host_loss(loss_count);
        if (const cudaError_t status = cudaMemcpy(host_loss.data(), loss_values, static_cast<std::size_t>(loss_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy loss values failed: "} + cudaGetErrorString(status)};
        for (const float loss : host_loss) out_loss_sum += loss;
    }

} // namespace hyfluid::cuda
