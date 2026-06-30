#ifndef HYFLUID_TRAIN_H
#define HYFLUID_TRAIN_H

#include "hyfluid.train.config.h"
#include <cstdint>
#include <type_traits>

namespace hyfluid::cuda {
    // Device memory.
    void free_device_buffers(void** pointers, std::size_t count) noexcept;

    template <typename... Pointers>
        requires ((std::is_pointer_v<Pointers> && ...))
    void free_device_buffers(Pointers&... pointers) noexcept {
        if constexpr (sizeof...(Pointers) > 0u) {
            void* raw[] = {const_cast<void*>(static_cast<const void*>(pointers))...};
            free_device_buffers(raw, sizeof...(Pointers));
            ((pointers = nullptr), ...);
        }
    }

    // Dataset.
    void upload_dataset(const std::uint8_t* pixels, std::size_t pixel_bytes, const float* camera, std::size_t camera_count, const float* intrinsics, std::size_t intrinsics_count, const float* times, std::size_t time_count, const std::uint32_t* view_indices, std::size_t view_index_count, const std::uint32_t* time_indices, std::size_t time_index_count, const std::uint32_t* frame_indices, std::size_t frame_index_count, const std::uint8_t*& out_pixels, const float*& out_camera, const float*& out_intrinsics, const float*& out_times, const std::uint32_t*& out_view_indices, const std::uint32_t*& out_time_indices, const std::uint32_t*& out_frame_indices);
    void upload_field_domain(const float* field_to_world_linear, float*& out_field_to_world_linear);

    // Sampler.
    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy, std::uint32_t*& out_occupancy_grid_occupied_count);
    void set_occupancy_grid_full(std::uint8_t* occupancy, std::uint32_t* occupancy_grid_occupied_count);
    void sample_training_batch(const float* camera, const float* intrinsics, const std::uint32_t* frame_indices, const float* field_to_world_linear, std::uint32_t view_count, std::uint32_t time_count, std::uint32_t width, std::uint32_t height, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);

    // Loss and compaction.
    void allocate_training_loss_buffers(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values, std::uint16_t*& out_network_output_gradients);
    void compute_training_loss_and_compact_samples(std::uint32_t rays_per_batch, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, const std::uint32_t* frame_indices, std::uint32_t view_count, std::uint32_t time_count, std::uint32_t width, std::uint32_t height, const std::uint16_t* network_output, std::uint32_t* compacted_sample_counter, const std::uint32_t* ray_indices, std::uint32_t* numsteps, const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, float* param_gradients, const std::uint16_t* params, float* loss_values);
    void pad_compacted_training_batch(const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);

    // Evaluation.
    void allocate_evaluation_buffers(std::uint32_t render_pixel_capacity, std::uint32_t*& out_evaluation_numsteps, std::uint32_t*& out_evaluation_sample_counter, std::uint32_t*& out_evaluation_overflow_counter, double*& out_evaluation_loss_sum, std::uint8_t*& out_evaluation_pixels);
    void run_evaluation_image(const std::uint8_t* pixels, const float* camera, const float* intrinsics, const std::uint32_t* frame_indices, const float* field_to_world_linear, std::uint32_t view_count, std::uint32_t time_count, std::uint32_t width, std::uint32_t height, std::uint32_t view_index, std::uint32_t time_index, std::uint32_t render_pixel_capacity, const std::uint8_t* occupancy, const std::uint16_t* params, float* sample_coords, std::uint16_t* network_input, std::uint16_t* network_hidden, std::uint16_t* network_output, void* cublaslt_handle, std::uint8_t* cublaslt_workspace, std::uint32_t* evaluation_numsteps, std::uint32_t* evaluation_sample_counter, std::uint32_t* evaluation_overflow_counter, double* evaluation_loss_sum, std::uint8_t* evaluation_pixels, std::uint8_t* host_evaluation_pixels, double& out_loss_sum);

    // Network.
    void allocate_network_buffers(std::uint16_t*& out_network_input, std::uint16_t*& out_network_hidden, std::uint16_t*& out_network_output, std::uint16_t*& out_network_input_gradients, std::uint16_t*& out_network_hidden_gradients, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace);
    void destroy_network_handle(void*& cublaslt_handle) noexcept;
    void evaluate_network(std::uint32_t sample_count, const float* sample_coords, const std::uint16_t* params, std::uint16_t* network_input, std::uint16_t* network_hidden, std::uint16_t* network_output, void* cublaslt_handle, std::uint8_t* cublaslt_workspace);
    void forward_network(const float* sample_coords, const std::uint16_t* params, std::uint16_t* network_input, std::uint16_t* network_hidden, std::uint16_t* network_output, void* cublaslt_handle, std::uint8_t* cublaslt_workspace);
    void backward_network(const float* sample_coords, const std::uint16_t* params, float* param_gradients, const std::uint16_t* network_input, const std::uint16_t* network_hidden, const std::uint16_t* network_output, const std::uint16_t* network_output_gradients, std::uint16_t* network_input_gradients, std::uint16_t* network_hidden_gradients, void* cublaslt_handle, std::uint8_t* cublaslt_workspace);

    // Trainable parameters and optimizer.
    void allocate_trainable_parameter_buffers(float*& out_params_full_precision, std::uint16_t*& out_params, float*& out_param_gradients);
    void initialize_trainable_parameters(float* params_full_precision, std::uint16_t* params);
    void download_trainable_parameters(const float* params_full_precision, float* out_params_full_precision);
    void upload_trainable_parameters(const float* params_full_precision, float* out_params_full_precision, std::uint16_t* out_params, float* out_param_gradients, float* optimizer_first_moments, float* optimizer_second_moments, std::uint32_t* optimizer_param_steps);
    void allocate_optimizer_buffers(float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps);
    void step_optimizer(float* params_full_precision, std::uint16_t* params, const float* gradients, float* first_moments, float* second_moments, std::uint32_t* param_steps);

    // Host readback.
    void read_counter(const std::uint32_t* counter, std::uint32_t& out_value);
    void read_loss_sum(const float* loss_values, std::uint32_t loss_count, float& out_loss_sum);
} // namespace hyfluid::cuda

#endif // HYFLUID_TRAIN_H
