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

    export template <typename Video>
    concept DynamicVideoLike = requires(const Video& video) {
        { std::string_view{video.frame_set} } -> std::same_as<std::string_view>;
        { std::string_view{video.file_name} } -> std::same_as<std::string_view>;
        requires std::ranges::contiguous_range<decltype((video.camera))>;
        requires std::same_as<std::ranges::range_value_t<decltype((video.camera))>, float>;
        { video.width } -> std::convertible_to<std::uint32_t>;
        { video.height } -> std::convertible_to<std::uint32_t>;
        { video.frame_count } -> std::convertible_to<std::uint32_t>;
        { video.frame_rate } -> std::convertible_to<std::uint32_t>;
        { video.view_index } -> std::convertible_to<std::uint32_t>;
        { video.focal } -> std::convertible_to<float>;
    };

    export template <typename Dataset>
    concept DynamicDatasetLike = requires(const Dataset& dataset) {
        { static_cast<float>(dataset.scene_scale) } -> std::same_as<float>;
        { static_cast<float>(dataset.near) } -> std::same_as<float>;
        { static_cast<float>(dataset.far) } -> std::same_as<float>;
        { static_cast<float>(dataset.phi) } -> std::same_as<float>;
        { static_cast<char>(dataset.rotation_axis) } -> std::same_as<char>;
        requires std::ranges::input_range<decltype((dataset.frame_sets))>;
        requires DynamicFrameSetLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((dataset.frame_sets))>>>;
        requires std::ranges::input_range<decltype((dataset.videos))>;
        requires DynamicVideoLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((dataset.videos))>>>;
        requires std::ranges::contiguous_range<decltype((dataset.sim_to_world))>;
        requires std::ranges::contiguous_range<decltype((dataset.world_to_sim))>;
        requires std::ranges::contiguous_range<decltype((dataset.voxel_scale))>;
        requires std::ranges::contiguous_range<decltype((dataset.render_center))>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.sim_to_world))>, float>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.world_to_sim))>, float>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.voxel_scale))>, float>;
        requires std::same_as<std::ranges::range_value_t<decltype((dataset.render_center))>, float>;
    };

    export struct FrameView final {
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

    export struct FrameSetView final {
        std::string_view name;
        std::span<const FrameView> frames;
        std::uint32_t view_count = 0u;
        std::uint32_t time_count = 0u;
    };

    export struct VideoView final {
        std::string_view frame_set;
        std::string_view file_name;
        std::span<const float> camera;
        std::uint32_t width       = 0u;
        std::uint32_t height      = 0u;
        std::uint32_t frame_count = 0u;
        std::uint32_t frame_rate  = 0u;
        std::uint32_t view_index  = 0u;
        float focal               = 0.0f;
    };

    export struct DatasetView final {
        std::span<const FrameSetView> frame_sets;
        std::span<const VideoView> videos;
        std::span<const float> sim_to_world;
        std::span<const float> world_to_sim;
        std::span<const float> voxel_scale;
        std::span<const float> render_center;
        float scene_scale  = 0.0f;
        float near         = 0.0f;
        float far          = 0.0f;
        float phi          = 0.0f;
        char rotation_axis = 'Y';
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
        std::uint32_t step                          = 0u;
        std::uint32_t rays_per_batch                = 0u;
        std::uint32_t ray_count                     = 0u;
        std::uint32_t sample_count                  = 0u;
        std::uint32_t occupancy_grid_occupied_cells = 0u;
        float loss                                  = 0.0f;
        float elapsed_ms                            = 0.0f;
        float sample_efficiency_ratio               = 0.0f;
        float occupancy_grid_ratio                  = 0.0f;
    };

    export struct EvaluationStats final {
        std::string frame_set;
        std::uint32_t step        = 0u;
        std::uint32_t image_count = 0u;
        std::uint64_t pixel_count = 0u;
        float mse                 = 0.0f;
        float psnr                = 0.0f;
        float elapsed_ms          = 0.0f;
        std::filesystem::path output_dir;
    };

    export struct HyFluid final {
        template <DynamicDatasetLike Dataset>
        explicit HyFluid(const Dataset& dataset) {
            std::vector<std::string_view> frame_set_names;
            std::vector<std::vector<FrameView>> frame_views_by_set;
            std::vector<VideoView> video_views;

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
                        .time = static_cast<float>(frame.time),
                        .view_index = static_cast<std::uint32_t>(frame.view_index),
                        .time_index = static_cast<std::uint32_t>(frame.time_index),
                    });
                }
            }

            for (const auto& video : dataset.videos) {
                video_views.push_back(VideoView{
                    .frame_set = std::string_view{video.frame_set},
                    .file_name = std::string_view{video.file_name},
                    .camera = std::span<const float>{std::ranges::data(video.camera), std::ranges::size(video.camera)},
                    .width = static_cast<std::uint32_t>(video.width),
                    .height = static_cast<std::uint32_t>(video.height),
                    .frame_count = static_cast<std::uint32_t>(video.frame_count),
                    .frame_rate = static_cast<std::uint32_t>(video.frame_rate),
                    .view_index = static_cast<std::uint32_t>(video.view_index),
                    .focal = static_cast<float>(video.focal),
                });
            }

            std::vector<FrameSetView> frame_set_views;
            frame_set_views.reserve(frame_views_by_set.size());
            std::size_t frame_set_index = 0uz;
            for (const auto& frame_set : dataset.frame_sets) {
                const std::vector<FrameView>& frame_views = frame_views_by_set.at(frame_set_index);
                frame_set_views.push_back(FrameSetView{
                    .name = frame_set_names.at(frame_set_index),
                    .frames = std::span<const FrameView>{frame_views.data(), frame_views.size()},
                    .view_count = static_cast<std::uint32_t>(frame_set.view_count),
                    .time_count = static_cast<std::uint32_t>(frame_set.time_count),
                });
                ++frame_set_index;
            }

            this->initialize(DatasetView{
                .frame_sets = std::span<const FrameSetView>{frame_set_views.data(), frame_set_views.size()},
                .videos = std::span<const VideoView>{video_views.data(), video_views.size()},
                .sim_to_world = std::span<const float>{std::ranges::data(dataset.sim_to_world), std::ranges::size(dataset.sim_to_world)},
                .world_to_sim = std::span<const float>{std::ranges::data(dataset.world_to_sim), std::ranges::size(dataset.world_to_sim)},
                .voxel_scale = std::span<const float>{std::ranges::data(dataset.voxel_scale), std::ranges::size(dataset.voxel_scale)},
                .render_center = std::span<const float>{std::ranges::data(dataset.render_center), std::ranges::size(dataset.render_center)},
                .scene_scale = static_cast<float>(dataset.scene_scale),
                .near = static_cast<float>(dataset.near),
                .far = static_cast<float>(dataset.far),
                .phi = static_cast<float>(dataset.phi),
                .rotation_axis = static_cast<char>(dataset.rotation_axis),
            });
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

        void initialize(const DatasetView& dataset);

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

        struct HostVideo final {
            std::string frame_set;
            std::string file_name;
            std::array<float, 12u> camera  = {};
            std::uint32_t frame_set_index = 0u;
            std::uint32_t width           = 0u;
            std::uint32_t height          = 0u;
            std::uint32_t frame_count     = 0u;
            std::uint32_t frame_rate      = 0u;
            std::uint32_t view_index      = 0u;
            float focal                   = 0.0f;
        };

        struct HostData {
            // Stable after construction: dynamic dataset metadata.
            std::vector<HostFrameSet> frame_sets = {};
            std::vector<HostFrame> frames        = {};
            std::vector<HostVideo> videos        = {};
            std::array<float, 16u> sim_to_world  = {};
            std::array<float, 16u> world_to_sim  = {};
            std::array<float, 3u> voxel_scale    = {};
            std::array<float, 3u> render_center  = {};
            float scene_scale                    = 0.0f;
            float near                           = 0.0f;
            float far                            = 0.0f;
            float phi                            = 0.0f;
            char rotation_axis                   = 'Y';

            // Mutated by optimize(): sampler step and latest counters.
            std::uint32_t current_step                    = 0u;
            std::uint32_t rays_per_batch                  = 0u;
            std::uint32_t measured_sample_count           = 0u;
            std::uint32_t occupancy_grid_occupied_cells   = 0u;
            std::uint64_t occupancy_grid_revision         = 0u;
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

            // Sampler.
            std::uint8_t* occupancy                    = nullptr;
            float* sample_coords                       = nullptr;
            float* rays                                = nullptr;
            std::uint32_t* ray_indices                 = nullptr;
            std::uint32_t* numsteps                    = nullptr;
            std::uint32_t* ray_counter                 = nullptr;
            std::uint32_t* sample_counter              = nullptr;
            std::uint32_t* occupancy_grid_occupied_count = nullptr;
        } device;
    };
} // namespace hyfluid::train
