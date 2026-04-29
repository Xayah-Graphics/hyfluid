#ifndef HYFLUID_TRAIN_H
#define HYFLUID_TRAIN_H

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define HYFLUID_CUDA_HOST_DEVICE __host__ __device__
#else
#define HYFLUID_CUDA_HOST_DEVICE
#endif

namespace hyfluid::cuda {
    inline constexpr std::uint32_t HASH_LEVELS             = 16u;
    inline constexpr std::uint32_t HASH_FEATURES_PER_LEVEL = 2u;
    inline constexpr std::uint32_t HASH_INPUT_WIDTH        = HASH_LEVELS * HASH_FEATURES_PER_LEVEL;
    inline constexpr std::uint32_t MLP_HIDDEN_WIDTH        = 64u;
    inline constexpr std::uint32_t DEFAULT_SAMPLES_PER_RAY = 192u;
    inline constexpr std::uint32_t DEFAULT_RAYS_PER_STEP   = 1024u;
    inline constexpr std::uint32_t TIME_OCCUPANCY_BINS     = 32u;
    inline constexpr std::uint32_t OCCUPANCY_GRID_SIZE     = 128u;
    inline constexpr std::uint32_t OCCUPANCY_GRID_CELLS    = OCCUPANCY_GRID_SIZE * OCCUPANCY_GRID_SIZE * OCCUPANCY_GRID_SIZE;
    inline constexpr std::uint64_t TRAIN_SEED              = 1337ull;

    struct Pcg32 final {
        std::uint64_t state = 0x853c49e6748fea9bULL;
        std::uint64_t inc   = 0xda3e39cb94b95bdbULL;

        Pcg32() = default;

        HYFLUID_CUDA_HOST_DEVICE explicit Pcg32(const std::uint64_t initstate, const std::uint64_t initseq = 1ull) {
            this->seed(initstate, initseq);
        }

        HYFLUID_CUDA_HOST_DEVICE void seed(const std::uint64_t initstate, const std::uint64_t initseq) {
            this->state = 0ull;
            this->inc   = (initseq << 1ull) | 1ull;
            this->next_uint();
            this->state += initstate;
            this->next_uint();
        }

        HYFLUID_CUDA_HOST_DEVICE std::uint32_t next_uint() {
            const std::uint64_t oldstate = this->state;
            this->state                  = oldstate * 0x5851f42d4c957f2dULL + this->inc;
            const std::uint32_t xorshift = static_cast<std::uint32_t>(((oldstate >> 18ull) ^ oldstate) >> 27ull);
            const std::uint32_t rot      = static_cast<std::uint32_t>(oldstate >> 59ull);
            return (xorshift >> rot) | (xorshift << ((~rot + 1u) & 31u));
        }

        HYFLUID_CUDA_HOST_DEVICE float next_float() {
            union {
                std::uint32_t bits;
                float value;
            } result    = {};
            result.bits = (this->next_uint() >> 9u) | 0x3f800000u;
            return result.value - 1.0f;
        }

        HYFLUID_CUDA_HOST_DEVICE void advance(std::uint64_t delta) {
            std::uint64_t cur_mult = 0x5851f42d4c957f2dULL;
            std::uint64_t cur_plus = this->inc;
            std::uint64_t acc_mult = 1ull;
            std::uint64_t acc_plus = 0ull;
            while (delta > 0ull) {
                if ((delta & 1ull) != 0ull) {
                    acc_mult *= cur_mult;
                    acc_plus = acc_plus * cur_mult + cur_plus;
                }
                cur_plus = (cur_mult + 1ull) * cur_plus;
                cur_mult *= cur_mult;
                delta >>= 1ull;
            }
            this->state = acc_mult * this->state + acc_plus;
        }
    };

#undef HYFLUID_CUDA_HOST_DEVICE

    void free_device_buffers(void** pointers, std::size_t count) noexcept;
    void upload_bytes(const std::uint8_t* data, std::size_t byte_count, const std::uint8_t*& out_data);
    void upload_floats(const float* data, std::size_t value_count, const float*& out_data);
    void upload_uint32s(const std::uint32_t* data, std::size_t value_count, const std::uint32_t*& out_data);
    void allocate_float_buffer(std::size_t value_count, float*& out_data);
    void allocate_double_buffer(std::size_t value_count, double*& out_data);
    void allocate_byte_buffer(std::size_t byte_count, std::uint8_t*& out_data);
    void allocate_uint32_buffer(std::size_t value_count, std::uint32_t*& out_data);
    void initialize_parameters(std::uint32_t param_count, std::uint32_t hash_param_count, std::uint32_t w1_offset, std::uint32_t w2_offset, std::uint32_t color_offset, float* params, float* gradients, float* first_moments, float* second_moments);
    void initialize_occupancy(std::uint8_t* occupancy_bits, float* occupancy_values, std::uint32_t* occupancy_counts);
    void update_time_occupancy(std::uint32_t time_bin, const float* sim_to_world, const float* voxel_scale, const std::uint32_t* hash_offsets, const std::uint32_t* hash_entries, const std::uint32_t* hash_resolutions, const std::uint32_t* hash_dense, std::uint32_t w1_offset, std::uint32_t w2_offset, const float* params, std::uint8_t* occupancy_bits, float* occupancy_values, std::uint32_t* occupancy_counts);
    void run_training_step(const std::uint8_t* train_pixels, const float* train_cameras, std::uint32_t train_view_count, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, const float* world_to_sim, const float* voxel_scale, float near_plane, float far_plane, std::uint32_t samples_per_ray, std::uint32_t rays_per_step, std::uint32_t current_step, const std::uint32_t* hash_offsets, const std::uint32_t* hash_entries, const std::uint32_t* hash_resolutions, const std::uint32_t* hash_dense, std::uint32_t hash_param_count, std::uint32_t w1_offset, std::uint32_t w2_offset, std::uint32_t color_offset, const std::uint8_t* occupancy_bits, float* params, float* gradients, float* loss_values, std::uint32_t* skipped_sample_counter);
    void step_radam(std::uint32_t param_count, std::uint32_t hash_param_count, std::uint32_t current_step, float learning_rate, float* params, float* gradients, float* first_moments, float* second_moments);
    void evaluate_test_frame(const std::uint8_t* test_pixels, const float* test_cameras, std::uint32_t test_view_index, std::uint32_t frame_index, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, const float* world_to_sim, const float* voxel_scale, float near_plane, float far_plane, std::uint32_t samples_per_ray, const std::uint32_t* hash_offsets, const std::uint32_t* hash_entries, const std::uint32_t* hash_resolutions, const std::uint32_t* hash_dense, std::uint32_t w1_offset, std::uint32_t w2_offset, std::uint32_t color_offset, const std::uint8_t* occupancy_bits, const float* params, double* loss_sum, std::uint8_t* comparison_pixels);
    void download_floats(const float* data, std::size_t value_count, float* out_data);
    void upload_parameters(std::uint32_t param_count, const float* data, float* params, float* gradients, float* first_moments, float* second_moments);
    void download_double(const double* data, double& out_value);
    void download_uint32(const std::uint32_t* data, std::uint32_t& out_value);
    void download_bytes(const std::uint8_t* data, std::size_t byte_count, std::uint8_t* out_data);
} // namespace hyfluid::cuda

#endif // HYFLUID_TRAIN_H
