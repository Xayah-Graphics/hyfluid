export module hyfluid.train;
import std;

namespace hyfluid::train {
    export template <typename Frame>
    concept DynamicFrameLike = requires(const Frame& frame) {
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
        { frame.time } -> std::convertible_to<float>;
        { frame.view_index } -> std::convertible_to<std::uint32_t>;
        { frame.time_index } -> std::convertible_to<std::uint32_t>;
    };

    export template <typename FrameSet>
    concept DynamicFrameSetLike = requires(const FrameSet& frame_set) {
        { std::string_view{frame_set.name} } -> std::same_as<std::string_view>;
        requires std::ranges::input_range<decltype((frame_set.frames))>;
        requires DynamicFrameLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((frame_set.frames))>>>;
        { frame_set.view_count } -> std::convertible_to<std::uint32_t>;
        { frame_set.time_count } -> std::convertible_to<std::uint32_t>;
    };

    export template <typename Dataset>
    concept DynamicDatasetLike = requires(const Dataset& dataset) {
        requires std::ranges::input_range<decltype((dataset.frame_sets))>;
        requires DynamicFrameSetLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((dataset.frame_sets))>>>;
        requires std::ranges::contiguous_range<decltype((dataset.sim_to_world))>;
        requires std::ranges::contiguous_range<decltype((dataset.voxel_scale))>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.sim_to_world))>, float>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.voxel_scale))>, float>;
    };

    export struct OptimizationRequest final {
        std::string_view frame_set;
        std::int32_t iterations = 1;
    };

    export struct EvaluationRequest final {
        std::string_view frame_set;
        std::filesystem::path output_dir;
    };

    export struct OptimizationStats final {
        std::uint32_t step                           = 0u;
        std::uint32_t rays_per_batch                 = 0u;
        std::uint32_t ray_count                      = 0u;
        std::uint32_t sample_count_before_compaction = 0u;
        std::uint32_t sample_count                   = 0u;
        std::uint32_t occupancy_grid_occupied_cells  = 0u;
        float loss                                   = 0.0f;
        float psnr                                   = 0.0f;
        float elapsed_ms                             = 0.0f;
        float sample_efficiency_ratio                = 0.0f;
        float occupancy_grid_ratio                   = 0.0f;
    };

    export struct EvaluationStats final {
        std::string frame_set;
        std::uint32_t step                 = 0u;
        std::uint32_t render_width         = 0u;
        std::uint32_t render_height        = 0u;
        std::uint32_t image_count          = 0u;
        std::uint32_t rendered_image_count = 0u;
        std::uint64_t pixel_count          = 0u;
        float mse                          = 0.0f;
        float psnr                         = 0.0f;
        float elapsed_ms                   = 0.0f;
        std::filesystem::path output_dir;
    };

    struct FrameView final {
        std::span<const std::uint8_t> rgba;
        std::span<const float> camera;
        std::uint32_t width      = 0u;
        std::uint32_t height     = 0u;
        float focal_x            = 0.0f;
        float focal_y            = 0.0f;
        float principal_x        = 0.0f;
        float principal_y        = 0.0f;
        float time               = 0.0f;
        std::uint32_t view_index = 0u;
        std::uint32_t time_index = 0u;
    };

    struct FrameSetView final {
        std::string_view name;
        std::span<const FrameView> frames;
        std::uint32_t view_count = 0u;
        std::uint32_t time_count = 0u;
    };

    export struct HyFluid final {
        template <DynamicDatasetLike Dataset>
        explicit HyFluid(const Dataset& dataset) {
            std::vector<std::vector<FrameView>> frame_views_by_set;
            std::vector<FrameSetView> frame_set_views;

            for (const auto& frame_set : dataset.frame_sets) {
                std::vector<FrameView>& frame_views = frame_views_by_set.emplace_back();
                for (const auto& frame : frame_set.frames) {
                    frame_views.push_back(FrameView{
                        .rgba        = std::span<const std::uint8_t>{std::ranges::data(frame.rgba), std::ranges::size(frame.rgba)},
                        .camera      = std::span<const float>{std::ranges::data(frame.camera), std::ranges::size(frame.camera)},
                        .width       = static_cast<std::uint32_t>(frame.width),
                        .height      = static_cast<std::uint32_t>(frame.height),
                        .focal_x     = static_cast<float>(frame.focal_x),
                        .focal_y     = static_cast<float>(frame.focal_y),
                        .principal_x = static_cast<float>(frame.principal_x),
                        .principal_y = static_cast<float>(frame.principal_y),
                        .time        = static_cast<float>(frame.time),
                        .view_index  = static_cast<std::uint32_t>(frame.view_index),
                        .time_index  = static_cast<std::uint32_t>(frame.time_index),
                    });
                }
                frame_set_views.push_back(FrameSetView{
                    .name       = std::string_view{frame_set.name},
                    .frames     = std::span<const FrameView>{frame_views.data(), frame_views.size()},
                    .view_count = static_cast<std::uint32_t>(frame_set.view_count),
                    .time_count = static_cast<std::uint32_t>(frame_set.time_count),
                });
            }

            this->initialize(std::span<const FrameSetView>{frame_set_views.data(), frame_set_views.size()}, std::span<const float>{std::ranges::data(dataset.sim_to_world), std::ranges::size(dataset.sim_to_world)}, std::span<const float>{std::ranges::data(dataset.voxel_scale), std::ranges::size(dataset.voxel_scale)});
        }

        ~HyFluid() noexcept;
        HyFluid(const HyFluid&)                = delete;
        HyFluid& operator=(const HyFluid&)     = delete;
        HyFluid(HyFluid&&) noexcept            = delete;
        HyFluid& operator=(HyFluid&&) noexcept = delete;

        [[nodiscard]] std::expected<OptimizationStats, std::string> optimize(const OptimizationRequest& request);
        [[nodiscard]] std::expected<EvaluationStats, std::string> evaluate(const EvaluationRequest& request) const;
        [[nodiscard]] std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        [[nodiscard]] std::expected<void, std::string> load_weights(const std::filesystem::path& path) const;

    private:
        void initialize(std::span<const FrameSetView> frame_sets, std::span<const float> sim_to_world, std::span<const float> voxel_scale);

    public:
        struct HostFrameSet final {
            std::string name;
            std::uint32_t frame_offset      = 0u;
            std::uint32_t frame_count       = 0u;
            std::uint32_t view_count        = 0u;
            std::uint32_t time_count        = 0u;
            std::uint32_t width             = 0u;
            std::uint32_t height            = 0u;
            std::uint64_t pixel_bytes       = 0u;
            std::uint64_t camera_values     = 0u;
            std::uint64_t intrinsics_values = 0u;
        };

        struct HostFrame final {
            std::uint32_t frame_set_index = 0u;
            std::uint32_t width           = 0u;
            std::uint32_t height          = 0u;
            float focal_x                 = 0.0f;
            float focal_y                 = 0.0f;
            float principal_x             = 0.0f;
            float principal_y             = 0.0f;
            float time                    = 0.0f;
            std::uint32_t view_index      = 0u;
            std::uint32_t time_index      = 0u;
            std::uint64_t pixel_offset    = 0u;
        };

        struct HostData {
            // Stable after construction: dynamic dataset metadata.
            std::vector<HostFrameSet> frame_sets        = {};
            std::vector<HostFrame> frames               = {};
            std::array<float, 9u> field_to_world_linear = {};

            // Mutated by optimize(): training step, adaptive batch shape, and latest counters.
            std::uint32_t current_step                            = 0u;
            std::uint32_t rays_per_batch                          = 0u;
            std::uint32_t inference_sample_count                  = 0u;
            std::uint32_t evaluation_pixel_capacity               = 0u;
            std::uint32_t measured_sample_count_before_compaction = 0u;
            std::uint32_t measured_sample_count                   = 0u;
            std::uint32_t occupancy_grid_occupied_cells           = 0u;
        } host;

        struct DeviceFrameSet final {
            const std::uint8_t* pixels         = nullptr;
            const float* camera                = nullptr;
            const float* intrinsics            = nullptr;
            const float* times                 = nullptr;
            const std::uint32_t* view_indices  = nullptr;
            const std::uint32_t* time_indices  = nullptr;
            const std::uint32_t* frame_indices = nullptr;
        };

        struct DeviceData {
            // Dataset.
            std::vector<DeviceFrameSet> frame_sets = {};
            float* field_to_world_linear           = nullptr;

            // Sampler.
            std::uint8_t* occupancy                      = nullptr;
            float* sample_coords                         = nullptr;
            float* rays                                  = nullptr;
            std::uint32_t* ray_indices                   = nullptr;
            std::uint32_t* numsteps                      = nullptr;
            std::uint32_t* ray_counter                   = nullptr;
            std::uint32_t* sample_counter                = nullptr;
            std::uint32_t* occupancy_grid_occupied_count = nullptr;

            // Loss and compaction.
            std::uint32_t* compacted_sample_counter = nullptr;
            float* compacted_sample_coords          = nullptr;
            float* loss_values                      = nullptr;
            std::uint16_t* network_output_gradients = nullptr;

            // Network.
            std::uint16_t* network_input            = nullptr;
            std::uint16_t* network_hidden           = nullptr;
            std::uint16_t* network_output           = nullptr;
            std::uint16_t* network_input_gradients  = nullptr;
            std::uint16_t* network_hidden_gradients = nullptr;
            void* cublaslt_handle                   = nullptr;
            std::uint8_t* cublaslt_workspace        = nullptr;

            // Trainable parameters.
            float* params_full_precision = nullptr;
            std::uint16_t* params        = nullptr;
            float* param_gradients       = nullptr;

            // Optimizer.
            float* optimizer_first_moments       = nullptr;
            float* optimizer_second_moments      = nullptr;
            std::uint32_t* optimizer_param_steps = nullptr;

            // Evaluation.
            std::uint32_t* evaluation_numsteps         = nullptr;
            std::uint32_t* evaluation_sample_counter   = nullptr;
            std::uint32_t* evaluation_overflow_counter = nullptr;
            double* evaluation_loss_sum                = nullptr;
            std::uint8_t* evaluation_pixels            = nullptr;
        } device;
    };
} // namespace hyfluid::train
