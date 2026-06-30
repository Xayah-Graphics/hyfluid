export module hyfluid.inspector;
import std;
import hyfluid.train;

namespace hyfluid::inspector {
    export inline constexpr std::uint32_t DensitySliceDimension = 128u;
    export inline constexpr std::uint64_t SamplerPointInstanceBytes   = static_cast<std::uint64_t>(8u * sizeof(float));
    export inline constexpr std::uint64_t SamplerSegmentInstanceBytes = static_cast<std::uint64_t>(12u * sizeof(float));

    export enum class DensitySliceEncoding : std::uint32_t {
        MortonFloat32 = 1u,
    };

    export enum class TrainingCoordinateSpace : std::uint32_t {
        Field = 0u,
    };

    export enum class OccupancyGridState : std::uint32_t {
        Full    = 0u,
        Static  = 1u,
        Learned = 2u,
    };

    export struct TrainingDomainView final {
        std::array<float, 3u> field_min{0.0f, 0.0f, 0.0f};
        std::array<float, 3u> field_max{1.0f, 1.0f, 1.0f};
        std::array<float, 3u> field_metric_extent{};
        TrainingCoordinateSpace coordinate_space{TrainingCoordinateSpace::Field};
        OccupancyGridState occupancy_state{OccupancyGridState::Full};
    };

    export struct TrainingBatchDiagnostics final {
        std::array<float, 3u> sample_coord_min{};
        std::array<float, 3u> sample_coord_max{};
        float time_min = 0.0f;
        float time_max = 0.0f;
        float dt_metric_min = 0.0f;
        float dt_metric_mean = 0.0f;
        float dt_metric_max = 0.0f;
        float metric_per_field_unit_min = 0.0f;
        float metric_per_field_unit_mean = 0.0f;
        float metric_per_field_unit_max = 0.0f;
    };

    export struct TrainingModelDiagnostics final {
        float global_rgb_param = 0.0f;
        float global_rgb_color = 0.0f;
        float global_rgb_gradient = 0.0f;
    };

    export struct DensitySliceSampleRequest final {
        std::array<std::uint32_t, 3u> dimensions{DensitySliceDimension, DensitySliceDimension, DensitySliceDimension};
        std::uint32_t time_count{};
        std::uint32_t time_index{};
        float* output_density{};
        std::uint64_t byte_size{};
        DensitySliceEncoding encoding{DensitySliceEncoding::MortonFloat32};
    };

    export struct DensitySliceSampleStats final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t byte_size{};
        std::uint64_t revision{};
        float density_min = 0.0f;
        float density_max = 0.0f;
        float density_mean = 0.0f;
        std::uint64_t density_nonzero_count = 0u;
        DensitySliceEncoding encoding{DensitySliceEncoding::MortonFloat32};
    };

    export enum class OccupancyGridEncoding : std::uint32_t {
        MortonBitfield = 0u,
    };

    export struct OccupancyGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t cell_count = 0u;
        const std::uint8_t* bitfield{};
        std::uint64_t bitfield_bytes = 0u;
        std::uint32_t occupied_cells = 0u;
        std::uint64_t revision = 0u;
        OccupancyGridEncoding encoding{OccupancyGridEncoding::MortonBitfield};
        bool initialized = false;
    };

    export struct SamplerBatchDeviceView final {
        std::uint32_t current_step{};
        std::uint32_t ray_count{};
        std::uint32_t sample_count{};
        const float* rays{};
        const std::uint32_t* numsteps{};
        const float* sample_coords{};
        std::uint64_t revision{};
        bool initialized{};
    };

    export struct SamplerVisualizationRequest final {
        std::byte* point_instances{};
        std::uint64_t point_byte_size{};
        std::byte* segment_instances{};
        std::uint64_t segment_byte_size{};
        std::uint32_t time_count{};
        std::uint32_t time_index{};
        float point_radius{0.002f};
        float ray_width{1.5f};
        std::uint32_t width_mode{};
    };

    export struct SamplerVisualizationStats final {
        std::uint32_t ray_count{};
        std::uint32_t point_count{};
        std::uint64_t point_byte_size{};
        std::uint64_t segment_byte_size{};
        std::uint64_t revision{};
    };

    export struct Inspector final {
        explicit Inspector(const train::HyFluid& trainer);

        [[nodiscard]] TrainingDomainView training_domain_view() const;
        [[nodiscard]] TrainingBatchDiagnostics training_batch_diagnostics() const;
        [[nodiscard]] TrainingModelDiagnostics training_model_diagnostics() const;
        [[nodiscard]] SamplerBatchDeviceView sampler_batch_device_view() const;
        [[nodiscard]] OccupancyGridDeviceView occupancy_grid_device_view() const;
        [[nodiscard]] std::expected<DensitySliceSampleStats, std::string> sample_density_slice(DensitySliceSampleRequest request) const;
        [[nodiscard]] std::expected<SamplerVisualizationStats, std::string> write_sampler_visualization(SamplerVisualizationRequest request) const;

        const train::HyFluid* trainer = nullptr;
    };
} // namespace hyfluid::inspector
