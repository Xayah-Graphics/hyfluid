export module hyfluid.inspector;
import std;
import hyfluid.train;

namespace hyfluid::inspector {
    export inline constexpr std::uint64_t SamplerPointInstanceBytes   = static_cast<std::uint64_t>(8u * sizeof(float));
    export inline constexpr std::uint64_t SamplerSegmentInstanceBytes = static_cast<std::uint64_t>(12u * sizeof(float));

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

        [[nodiscard]] SamplerBatchDeviceView sampler_batch_device_view() const;
        [[nodiscard]] std::expected<SamplerVisualizationStats, std::string> write_sampler_visualization(SamplerVisualizationRequest request) const;

        const train::HyFluid* trainer = nullptr;
    };
} // namespace hyfluid::inspector
