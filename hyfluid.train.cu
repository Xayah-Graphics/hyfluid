#include "hyfluid.train.h"
#include <cuda_runtime.h>
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

        auto free_bytes  = static_cast<std::size_t>(0u);
        auto total_bytes = static_cast<std::size_t>(0u);
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
} // namespace hyfluid::cuda
