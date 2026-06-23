#include "hyfluid.train.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hyfluid::cuda {
    void free_dataset(std::uint8_t*& pixels, float*& camera) noexcept {
        if (pixels != nullptr) cudaFree(pixels);
        if (camera != nullptr) cudaFree(camera);
        pixels = nullptr;
        camera = nullptr;
    }

    void upload_dataset(const std::uint8_t* const pixels, const std::size_t pixels_bytes, const float* const camera, const std::size_t camera_count, std::uint8_t*& out_pixels, float*& out_camera) {
        out_pixels = nullptr;
        out_camera = nullptr;
        if (pixels == nullptr || camera == nullptr || pixels_bytes == 0u || camera_count == 0u) throw std::runtime_error{"invalid dataset upload input."};
        if (camera_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"camera upload is too large."};

        const std::size_t camera_bytes = camera_count * sizeof(float);
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&out_pixels), pixels_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc dataset pixels failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&out_camera), camera_bytes); status != cudaSuccess) {
            free_dataset(out_pixels, out_camera);
            throw std::runtime_error{std::string{"cudaMalloc dataset camera failed: "} + cudaGetErrorString(status)};
        }
        if (const cudaError_t status = cudaMemcpy(out_pixels, pixels, pixels_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) {
            free_dataset(out_pixels, out_camera);
            throw std::runtime_error{std::string{"cudaMemcpy dataset pixels failed: "} + cudaGetErrorString(status)};
        }
        if (const cudaError_t status = cudaMemcpy(out_camera, camera, camera_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) {
            free_dataset(out_pixels, out_camera);
            throw std::runtime_error{std::string{"cudaMemcpy dataset camera failed: "} + cudaGetErrorString(status)};
        }
    }
} // namespace hyfluid::cuda
