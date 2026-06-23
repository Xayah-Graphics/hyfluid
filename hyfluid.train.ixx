export module hyfluid.train;
import std;

namespace hyfluid::train {
    export template <typename Frame>
    concept FrameLike =
        requires(const Frame& frame) {
            requires std::ranges::contiguous_range<decltype((frame.rgba))>;
            requires std::same_as<std::ranges::range_value_t<decltype((frame.rgba))>, std::uint8_t>;
            requires std::ranges::contiguous_range<decltype((frame.camera))>;
            requires std::same_as<std::ranges::range_value_t<decltype((frame.camera))>, float>;
            { frame.width } -> std::convertible_to<std::uint32_t>;
            { frame.height } -> std::convertible_to<std::uint32_t>;
            { frame.focal_x } -> std::convertible_to<float>;
            { frame.focal_y } -> std::convertible_to<float>;
            { frame.principal_x } -> std::convertible_to<float>;
            { frame.principal_y } -> std::convertible_to<float>;
        };

    export template <typename FrameSet>
    concept FrameSetLike =
        requires(const FrameSet& frame_set) {
            { std::string_view{frame_set.name} } -> std::same_as<std::string_view>;
            requires std::ranges::input_range<decltype((frame_set.frames))>;
            requires FrameLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((frame_set.frames))>>>;
        };

    export template <typename Dataset>
    concept DatasetLike =
        requires(const Dataset& dataset) {
            { static_cast<float>(dataset.scene_scale) } -> std::same_as<float>;
            requires std::ranges::input_range<decltype((dataset.frame_sets))>;
            requires FrameSetLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((dataset.frame_sets))>>>;
        };

    export struct FrameView final {
        std::span<const std::uint8_t> rgba;
        std::span<const float> camera;
        std::uint32_t width = 0u;
        std::uint32_t height = 0u;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float principal_x = 0.0f;
        float principal_y = 0.0f;
    };

    export struct FrameSetView final {
        std::string_view name;
        std::span<const FrameView> frames;
    };

    export struct OptimizationRequest final {
        std::string_view frame_set;
        std::int32_t iterations = 1;
    };

    export struct EvaluationRequest final {
        std::string_view frame_set;
        std::optional<std::filesystem::path> comparison_output_dir;
    };

    export struct OptimizationStats final {
        std::uint32_t step = 0u;
        float loss = 0.0f;
        float elapsed_ms = 0.0f;
    };

    export struct EvaluationStats final {
        std::string frame_set;
        std::uint32_t step = 0u;
        std::uint32_t image_count = 0u;
        std::uint64_t pixel_count = 0u;
        float mse = 0.0f;
        float psnr = 0.0f;
        float elapsed_ms = 0.0f;
        std::filesystem::path output_dir;
    };

    export struct HyFluid final {
        template <DatasetLike Dataset>
        explicit HyFluid(const Dataset& dataset) {
            std::vector<std::string_view> frame_set_names;
            std::vector<std::vector<FrameView>> frame_views_by_set;
            for (const auto& frame_set : dataset.frame_sets) {
                frame_set_names.push_back(std::string_view{frame_set.name});
                std::vector<FrameView>& frame_views = frame_views_by_set.emplace_back();
                for (const auto& frame : frame_set.frames) {
                    frame_views.push_back(FrameView{
                        .rgba = std::span<const std::uint8_t>{std::ranges::data(frame.rgba), std::ranges::size(frame.rgba)},
                        .camera = std::span<const float>{std::ranges::data(frame.camera), std::ranges::size(frame.camera)},
                        .width = static_cast<std::uint32_t>(frame.width),
                        .height = static_cast<std::uint32_t>(frame.height),
                        .focal_x = static_cast<float>(frame.focal_x),
                        .focal_y = static_cast<float>(frame.focal_y),
                        .principal_x = static_cast<float>(frame.principal_x),
                        .principal_y = static_cast<float>(frame.principal_y),
                    });
                }
            }

            std::vector<FrameSetView> frame_set_views;
            frame_set_views.reserve(frame_views_by_set.size());
            for (const std::size_t frame_set_index : std::views::iota(0uz, frame_views_by_set.size())) {
                frame_set_views.push_back(FrameSetView{
                    .name = frame_set_names[frame_set_index],
                    .frames = std::span<const FrameView>{frame_views_by_set[frame_set_index]},
                });
            }
            this->initialize(frame_set_views, static_cast<float>(dataset.scene_scale));
        }

        ~HyFluid() noexcept;
        HyFluid(const HyFluid&) = delete;
        HyFluid& operator=(const HyFluid&) = delete;
        HyFluid(HyFluid&&) noexcept = delete;
        HyFluid& operator=(HyFluid&&) noexcept = delete;

        std::expected<OptimizationStats, std::string> optimize(OptimizationRequest request);
        std::expected<EvaluationStats, std::string> evaluate(EvaluationRequest request) const;
        std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        std::expected<void, std::string> load_weights(const std::filesystem::path& path);

        void initialize(std::span<const FrameSetView> frame_sets, float scene_scale);

        struct HostFrameSet final {
            std::string name;
            std::uint32_t frame_count = 0u;
            std::uint32_t width = 0u;
            std::uint32_t height = 0u;
            float focal_x = 0.0f;
            float focal_y = 0.0f;
            float principal_x = 0.0f;
            float principal_y = 0.0f;
            std::uint64_t pixel_bytes = 0u;
            std::uint64_t camera_values = 0u;
        };

        struct HostData final {
            std::vector<HostFrameSet> frame_sets = {};
            float scene_scale = 0.0f;
            std::uint32_t current_step = 0u;
        } host;

        struct DeviceFrameSet final {
            std::uint8_t* pixels = nullptr;
            float* camera = nullptr;
        };

        struct DeviceData final {
            std::vector<DeviceFrameSet> frame_sets = {};
        } device;
    };
} // namespace hyfluid::train
