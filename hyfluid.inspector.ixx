export module hyfluid.inspector;
import std;
import hyfluid.train;

namespace hyfluid::inspector {
    export inline constexpr std::uint32_t DensitySliceDimension = 128u;

    export struct TrainingBatchDiagnostics final {
        std::array<float, 3u> sample_coord_min{};
        std::array<float, 3u> sample_coord_max{};
        float time_min                   = 0.0f;
        float time_max                   = 0.0f;
        float dt_metric_min              = 0.0f;
        float dt_metric_mean             = 0.0f;
        float dt_metric_max              = 0.0f;
        float metric_per_field_unit_min  = 0.0f;
        float metric_per_field_unit_mean = 0.0f;
        float metric_per_field_unit_max  = 0.0f;
    };

    export struct TrainingModelDiagnostics final {
        float global_rgb_param    = 0.0f;
        float global_rgb_color    = 0.0f;
        float global_rgb_gradient = 0.0f;
    };

    export struct DensitySliceSampleRequest final {
        std::array<std::uint32_t, 3u> dimensions{DensitySliceDimension, DensitySliceDimension, DensitySliceDimension};
        std::uint32_t time_count{};
        std::uint32_t time_index{};
        float* output_density{};
        std::uint64_t byte_size{};
    };

    export struct DensitySliceSampleStats final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t byte_size{};
        std::uint64_t revision{};
        float density_min                   = 0.0f;
        float density_max                   = 0.0f;
        float density_mean                  = 0.0f;
        std::uint64_t density_nonzero_count = 0u;
    };

    export struct OccupancyGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint32_t bin_index = 0u;
        std::uint32_t bin_count = 0u;
        std::uint64_t cell_count = 0u;
        const std::uint8_t* bitfield{};
        std::uint64_t bitfield_bytes = 0u;
        std::uint32_t occupied_cells = 0u;
        bool initialized = false;
    };

    export struct Inspector final {
        explicit Inspector(const train::HyFluid& trainer);

        [[nodiscard]] TrainingBatchDiagnostics training_batch_diagnostics() const;
        [[nodiscard]] TrainingModelDiagnostics training_model_diagnostics() const;
        [[nodiscard]] OccupancyGridDeviceView occupancy_grid_device_view(std::uint32_t bin_index) const;
        [[nodiscard]] std::expected<DensitySliceSampleStats, std::string> sample_density_slice(DensitySliceSampleRequest request) const;

        const train::HyFluid* trainer = nullptr;
    };
} // namespace hyfluid::inspector
