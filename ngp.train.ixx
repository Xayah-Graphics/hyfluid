module;
#include "ngp.train.h"
export module hyfluid.train;
import std;
import hyfluid.dataset;

namespace hyfluid::train {
    export struct TrainOptions final {
        std::uint32_t rays_per_step           = hyfluid::cuda::DEFAULT_RAYS_PER_STEP;
        std::uint32_t samples_per_ray         = hyfluid::cuda::DEFAULT_SAMPLES_PER_RAY;
        std::uint32_t test_frame_limit        = 1u;
        std::filesystem::path test_output_dir = "test";
        float learning_rate                   = 5e-4f;
    };

    export struct TrainStats final {
        std::uint32_t step                      = 0u;
        std::uint32_t rays_per_step             = 0u;
        std::uint32_t samples_per_ray           = 0u;
        std::uint32_t occupancy_bin             = 0u;
        std::uint32_t occupancy_occupied_cells  = 0u;
        std::uint32_t occupancy_skipped_samples = 0u;
        float loss                              = 0.0f;
        float occupancy_ratio                   = 0.0f;
        float occupancy_update_ms               = 0.0f;
        float elapsed_ms                        = 0.0f;
    };

    export struct TestStats final {
        std::uint32_t step                   = 0u;
        std::uint32_t image_count            = 0u;
        std::uint32_t comparison_image_count = 0u;
        std::uint64_t pixel_count            = 0u;
        float mse                            = 0.0f;
        float psnr                           = 0.0f;
        float elapsed_ms                     = 0.0f;
        std::filesystem::path output_dir;
    };

    export class HyFluidDensity final {
    public:
        explicit HyFluidDensity(const hyfluid::dataset::ScalarRealDataset& dataset, const TrainOptions& options = {});
        ~HyFluidDensity() noexcept;

        HyFluidDensity(const HyFluidDensity&)                = delete;
        HyFluidDensity& operator=(const HyFluidDensity&)     = delete;
        HyFluidDensity(HyFluidDensity&&) noexcept            = delete;
        HyFluidDensity& operator=(HyFluidDensity&&) noexcept = delete;

        std::expected<TrainStats, std::string> train(std::int32_t iters);
        std::expected<TestStats, std::string> test() const;
        std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        std::expected<void, std::string> load_weights(const std::filesystem::path& path);

    private:
        struct HostData final {
            TrainOptions options;
            std::uint32_t train_view_count     = 0u;
            std::uint32_t test_view_count      = 0u;
            std::uint32_t frame_count          = 0u;
            std::uint32_t width                = 0u;
            std::uint32_t height               = 0u;
            float near_plane                   = 0.0f;
            float far_plane                    = 0.0f;
            std::array<float, 16> sim_to_world = {};
            std::array<float, 16> world_to_sim = {};
            std::array<float, 3> voxel_scale   = {};

            std::array<std::uint32_t, hyfluid::cuda::HASH_LEVELS> hash_offsets          = {};
            std::array<std::uint32_t, hyfluid::cuda::HASH_LEVELS> hash_entries          = {};
            std::array<std::uint32_t, hyfluid::cuda::HASH_LEVELS * 4u> hash_resolutions = {};
            std::array<std::uint32_t, hyfluid::cuda::HASH_LEVELS> hash_dense            = {};
            std::uint32_t hash_param_count                                              = 0u;
            std::uint32_t w1_offset                                                     = 0u;
            std::uint32_t w2_offset                                                     = 0u;
            std::uint32_t color_offset                                                  = 0u;
            std::uint32_t param_count                                                   = 0u;
            std::uint32_t current_step                                                  = 0u;
            std::uint32_t last_occupancy_bin                                            = 0u;
            std::uint32_t last_occupancy_occupied_cells                                 = 0u;
            std::uint32_t last_occupancy_skipped_samples                                = 0u;
            float last_occupancy_update_ms                                              = 0.0f;
        } host;

        struct DeviceData final {
            const std::uint8_t* train_pixels         = nullptr;
            const std::uint8_t* test_pixels          = nullptr;
            const float* train_cameras               = nullptr;
            const float* test_cameras                = nullptr;
            const float* sim_to_world                = nullptr;
            const float* world_to_sim                = nullptr;
            const float* voxel_scale                 = nullptr;
            const std::uint32_t* hash_offsets        = nullptr;
            const std::uint32_t* hash_entries        = nullptr;
            const std::uint32_t* hash_resolutions    = nullptr;
            const std::uint32_t* hash_dense          = nullptr;
            float* params                            = nullptr;
            float* gradients                         = nullptr;
            float* first_moments                     = nullptr;
            float* second_moments                    = nullptr;
            float* loss_values                       = nullptr;
            double* test_loss_sum                    = nullptr;
            std::uint8_t* comparison_pixels          = nullptr;
            std::uint8_t* occupancy_bits             = nullptr;
            float* occupancy_values                  = nullptr;
            std::uint32_t* occupancy_counts          = nullptr;
            std::uint32_t* occupancy_skipped_samples = nullptr;
        } device;
    };
} // namespace hyfluid::train
