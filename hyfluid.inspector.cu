#include "hyfluid.inspector.h"
#include "hyfluid.train.config.h"
#include "hyfluid.train.h"
#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hyfluid::cuda {
    namespace {
        [[nodiscard]] __device__ std::uint32_t spread_morton3d_bits(std::uint32_t value) {
            value &= 0x000003ffu;
            value = (value | (value << 16u)) & 0x030000ffu;
            value = (value | (value << 8u)) & 0x0300f00fu;
            value = (value | (value << 4u)) & 0x030c30c3u;
            value = (value | (value << 2u)) & 0x09249249u;
            return value;
        }

        [[nodiscard]] __device__ std::uint32_t morton3d_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
            return spread_morton3d_bits(x) | (spread_morton3d_bits(y) << 1u) | (spread_morton3d_bits(z) << 2u);
        }

        __global__ void write_density_slice_coords_kernel(const std::uint32_t offset, const std::uint32_t count, const std::uint32_t dim_x, const std::uint32_t dim_y, const std::uint32_t dim_z, const float time, float* __restrict__ sample_coords) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= count) return;

            const std::uint32_t cell_index = offset + i;
            const std::uint32_t display_x  = cell_index % dim_x;
            const std::uint32_t display_y  = (cell_index / dim_x) % dim_y;
            const std::uint32_t display_z  = cell_index / (dim_x * dim_y);

            float* coord = sample_coords + static_cast<std::uint64_t>(i) * train::config::sample_coord_floats;
            coord[0u]    = (static_cast<float>(display_z) + 0.5f) / static_cast<float>(dim_z);
            coord[1u]    = (static_cast<float>(display_y) + 0.5f) / static_cast<float>(dim_y);
            coord[2u]    = (static_cast<float>(display_x) + 0.5f) / static_cast<float>(dim_x);
            coord[3u]    = time;
            coord[4u]    = 0.0f;
        }

        __global__ void copy_density_slice_output_kernel(const std::uint32_t offset, const std::uint32_t count, const std::uint32_t dim_x, const std::uint32_t dim_y, const std::uint16_t* __restrict__ network_output, float* __restrict__ output_density) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= count) return;

            const std::uint32_t cell_index = offset + i;
            const std::uint32_t display_x  = cell_index % dim_x;
            const std::uint32_t display_y  = (cell_index / dim_x) % dim_y;
            const std::uint32_t display_z  = cell_index / (dim_x * dim_y);
            const std::uint32_t morton     = morton3d_index(display_x, display_y, display_z);
            const float density            = __half2float(reinterpret_cast<const __half*>(network_output)[i]);
            output_density[morton]         = fmaxf(density, 0.0f);
        }

        __global__ void compute_density_slice_stats_kernel(const std::uint32_t cell_count, const float* __restrict__ output_density, float* __restrict__ block_min, float* __restrict__ block_max, float* __restrict__ block_sum, std::uint32_t* __restrict__ block_nonzero) {
            __shared__ float shared_min[train::config::threads_per_block];
            __shared__ float shared_max[train::config::threads_per_block];
            __shared__ float shared_sum[train::config::threads_per_block];
            __shared__ std::uint32_t shared_nonzero[train::config::threads_per_block];

            const std::uint32_t i       = threadIdx.x + blockIdx.x * blockDim.x;
            const bool valid            = i < cell_count;
            const float density         = valid ? output_density[i] : 0.0f;
            shared_min[threadIdx.x]     = valid ? density : FLT_MAX;
            shared_max[threadIdx.x]     = valid ? density : 0.0f;
            shared_sum[threadIdx.x]     = valid ? density : 0.0f;
            shared_nonzero[threadIdx.x] = valid && density > 0.0f ? 1u : 0u;
            __syncthreads();

            for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
                if (threadIdx.x < stride) {
                    shared_min[threadIdx.x] = fminf(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
                    shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
                    shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
                    shared_nonzero[threadIdx.x] += shared_nonzero[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0u) {
                block_min[blockIdx.x]     = shared_min[0u];
                block_max[blockIdx.x]     = shared_max[0u];
                block_sum[blockIdx.x]     = shared_sum[0u];
                block_nonzero[blockIdx.x] = shared_nonzero[0u];
            }
        }

    } // namespace

    void read_float_value(const float* const values, const std::uint32_t index, float& out_value) {
        out_value = 0.0f;
        if (values == nullptr) throw std::runtime_error{"float value buffer is null."};
        if (const cudaError_t status = cudaMemcpy(&out_value, values + index, sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy float value failed: "} + cudaGetErrorString(status)};
    }

    void read_training_batch_diagnostics(const float* const sample_coords, const std::uint32_t sample_count, TrainingBatchDiagnostics& out_diagnostics) {
        out_diagnostics = {};
        if (sample_count == 0u) return;
        if (sample_coords == nullptr) throw std::runtime_error{"sample coords are null."};
        const std::size_t value_count = static_cast<std::size_t>(sample_count) * train::config::sample_coord_floats;
        std::vector<float> host_coords(value_count);
        if (const cudaError_t status = cudaMemcpy(host_coords.data(), sample_coords, value_count * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy training batch diagnostics failed: "} + cudaGetErrorString(status)};

        out_diagnostics.coord_min[0u]             = std::numeric_limits<float>::infinity();
        out_diagnostics.coord_min[1u]             = std::numeric_limits<float>::infinity();
        out_diagnostics.coord_min[2u]             = std::numeric_limits<float>::infinity();
        out_diagnostics.coord_max[0u]             = -std::numeric_limits<float>::infinity();
        out_diagnostics.coord_max[1u]             = -std::numeric_limits<float>::infinity();
        out_diagnostics.coord_max[2u]             = -std::numeric_limits<float>::infinity();
        out_diagnostics.time_min                  = std::numeric_limits<float>::infinity();
        out_diagnostics.time_max                  = -std::numeric_limits<float>::infinity();
        out_diagnostics.dt_metric_min             = std::numeric_limits<float>::infinity();
        out_diagnostics.dt_metric_max             = 0.0f;
        out_diagnostics.metric_per_field_unit_min = std::numeric_limits<float>::infinity();
        out_diagnostics.metric_per_field_unit_max = 0.0f;

        double dt_sum     = 0.0;
        double metric_sum = 0.0;
        for (std::uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
            const float* coord = host_coords.data() + static_cast<std::uint64_t>(sample_index) * train::config::sample_coord_floats;
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                out_diagnostics.coord_min[axis] = std::min(out_diagnostics.coord_min[axis], coord[axis]);
                out_diagnostics.coord_max[axis] = std::max(out_diagnostics.coord_max[axis], coord[axis]);
            }
            out_diagnostics.time_min                  = std::min(out_diagnostics.time_min, coord[3u]);
            out_diagnostics.time_max                  = std::max(out_diagnostics.time_max, coord[3u]);
            out_diagnostics.dt_metric_min             = std::min(out_diagnostics.dt_metric_min, coord[4u]);
            out_diagnostics.dt_metric_max             = std::max(out_diagnostics.dt_metric_max, coord[4u]);
            const float metric_per_field_unit         = coord[4u] / train::config::training_ray_stepsize;
            out_diagnostics.metric_per_field_unit_min = std::min(out_diagnostics.metric_per_field_unit_min, metric_per_field_unit);
            out_diagnostics.metric_per_field_unit_max = std::max(out_diagnostics.metric_per_field_unit_max, metric_per_field_unit);
            dt_sum += static_cast<double>(coord[4u]);
            metric_sum += static_cast<double>(metric_per_field_unit);
        }
        out_diagnostics.dt_metric_mean             = static_cast<float>(dt_sum / static_cast<double>(sample_count));
        out_diagnostics.metric_per_field_unit_mean = static_cast<float>(metric_sum / static_cast<double>(sample_count));
    }

    void sample_density_slice(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z, float time, const std::uint16_t* params, float* sample_coords, std::uint16_t* network_input, std::uint16_t* network_hidden, std::uint16_t* network_output, void* cublaslt_handle, std::uint8_t* cublaslt_workspace, float* output_density, float& out_density_min, float& out_density_max, float& out_density_mean, std::uint64_t& out_density_nonzero_count) {
        if (dim_x != train::config::nerf_grid_size || dim_y != train::config::nerf_grid_size || dim_z != train::config::nerf_grid_size) throw std::runtime_error{"density slice dimensions must match HyFluid grid size."};
        if (time < 0.0f || time > 1.0f) throw std::runtime_error{"density slice time is outside [0, 1]."};
        if (params == nullptr || sample_coords == nullptr || network_input == nullptr || network_hidden == nullptr || network_output == nullptr || cublaslt_handle == nullptr || cublaslt_workspace == nullptr || output_density == nullptr) throw std::runtime_error{"invalid density slice input."};

        for (std::uint32_t offset = 0u; offset < train::config::nerf_grid_cells; offset += train::config::network_batch_size) {
            const std::uint32_t remaining = train::config::nerf_grid_cells - offset;
            const std::uint32_t count     = remaining < train::config::network_batch_size ? remaining : train::config::network_batch_size;
            const std::uint32_t blocks    = (count + train::config::threads_per_block - 1u) / train::config::threads_per_block;

            write_density_slice_coords_kernel<<<blocks, train::config::threads_per_block>>>(offset, count, dim_x, dim_y, dim_z, time, sample_coords);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"write_density_slice_coords_kernel failed: "} + cudaGetErrorString(status)};

            evaluate_network(count, sample_coords, params, network_input, network_hidden, network_output, cublaslt_handle, cublaslt_workspace);

            copy_density_slice_output_kernel<<<blocks, train::config::threads_per_block>>>(offset, count, dim_x, dim_y, network_output, output_density);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"copy_density_slice_output_kernel failed: "} + cudaGetErrorString(status)};
        }

        const std::uint32_t block_count = (train::config::nerf_grid_cells + train::config::threads_per_block - 1u) / train::config::threads_per_block;
        float* block_min                = nullptr;
        float* block_max                = nullptr;
        float* block_sum                = nullptr;
        std::uint32_t* block_nonzero    = nullptr;
        try {
            if (const cudaError_t status = cudaMalloc(&block_min, static_cast<std::size_t>(block_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density stats min failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(&block_max, static_cast<std::size_t>(block_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density stats max failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(&block_sum, static_cast<std::size_t>(block_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density stats sum failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(&block_nonzero, static_cast<std::size_t>(block_count) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density stats nonzero failed: "} + cudaGetErrorString(status)};
            compute_density_slice_stats_kernel<<<block_count, train::config::threads_per_block>>>(train::config::nerf_grid_cells, output_density, block_min, block_max, block_sum, block_nonzero);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_density_slice_stats_kernel failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after density slice sampling failed: "} + cudaGetErrorString(status)};

            std::vector<float> host_min(block_count);
            std::vector<float> host_max(block_count);
            std::vector<float> host_sum(block_count);
            std::vector<std::uint32_t> host_nonzero(block_count);
            if (const cudaError_t status = cudaMemcpy(host_min.data(), block_min, static_cast<std::size_t>(block_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy density stats min failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemcpy(host_max.data(), block_max, static_cast<std::size_t>(block_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy density stats max failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemcpy(host_sum.data(), block_sum, static_cast<std::size_t>(block_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy density stats sum failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemcpy(host_nonzero.data(), block_nonzero, static_cast<std::size_t>(block_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy density stats nonzero failed: "} + cudaGetErrorString(status)};

            double density_sum        = 0.0;
            out_density_min           = FLT_MAX;
            out_density_max           = 0.0f;
            out_density_nonzero_count = 0u;
            for (std::uint32_t block = 0u; block < block_count; ++block) {
                out_density_min = std::min(out_density_min, host_min[block]);
                out_density_max = std::max(out_density_max, host_max[block]);
                density_sum += static_cast<double>(host_sum[block]);
                out_density_nonzero_count += host_nonzero[block];
            }
            out_density_mean = static_cast<float>(density_sum / static_cast<double>(train::config::nerf_grid_cells));
            if (out_density_min == FLT_MAX) out_density_min = 0.0f;
            const cudaError_t free_min_status     = cudaFree(block_min);
            block_min                             = nullptr;
            const cudaError_t free_max_status     = cudaFree(block_max);
            block_max                             = nullptr;
            const cudaError_t free_sum_status     = cudaFree(block_sum);
            block_sum                             = nullptr;
            const cudaError_t free_nonzero_status = cudaFree(block_nonzero);
            block_nonzero                         = nullptr;
            if (free_min_status != cudaSuccess) throw std::runtime_error{std::string{"cudaFree density stats min failed: "} + cudaGetErrorString(free_min_status)};
            if (free_max_status != cudaSuccess) throw std::runtime_error{std::string{"cudaFree density stats max failed: "} + cudaGetErrorString(free_max_status)};
            if (free_sum_status != cudaSuccess) throw std::runtime_error{std::string{"cudaFree density stats sum failed: "} + cudaGetErrorString(free_sum_status)};
            if (free_nonzero_status != cudaSuccess) throw std::runtime_error{std::string{"cudaFree density stats nonzero failed: "} + cudaGetErrorString(free_nonzero_status)};
        } catch (...) {
            if (block_min != nullptr) cudaFree(block_min);
            if (block_max != nullptr) cudaFree(block_max);
            if (block_sum != nullptr) cudaFree(block_sum);
            if (block_nonzero != nullptr) cudaFree(block_nonzero);
            throw;
        }
    }

} // namespace hyfluid::cuda
