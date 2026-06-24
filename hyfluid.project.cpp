module;

#if defined(_WIN32)
#define SPECTRA_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_SCENE_EXPORT __attribute__((visibility("default")))
#endif

module hyfluid.project;

import dataset.scalar_real;
import hyfluid.plugin;
import std;

namespace hyfluid::project {
    namespace {
        constexpr char section_dataset_id[] = "dataset";
        constexpr char section_timeline_id[] = "timeline";
        constexpr char section_diagnostics_id[] = "diagnostics";

        struct Vector3 final {
            float x{};
            float y{};
            float z{};
        };

        struct CameraBasis final {
            Vector3 right{};
            Vector3 down{};
            Vector3 forward{};
        };

        struct DatasetOptions final {
            std::filesystem::path dataset_path;
            std::vector<std::string> frame_sets{"train"};
            float scene_scale{1.0f};
            std::uint64_t view_stride{1u};
            std::uint64_t max_views{};
        };

        struct FrameSetRuntime final {
            std::string name;
            std::uint32_t dataset_frame_set_index{};
            std::uint32_t view_count{};
            std::uint32_t time_count{};
            std::uint32_t width{};
            std::uint32_t height{};
            std::uint32_t frame_rate{};
            std::vector<std::uint32_t> frame_indices;
            std::vector<std::uint32_t> visible_views;
        };

        [[nodiscard]] float parse_float(const std::string& text, const std::string_view name) {
            float value{};
            const char* const begin = text.data();
            const char* const end = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value)) throw std::runtime_error{std::format("{} must be a finite float.", name)};
            return value;
        }

        [[nodiscard]] std::uint64_t parse_u64(const std::string& text, const std::string_view name) {
            std::uint64_t value{};
            const char* const begin = text.data();
            const char* const end = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end) throw std::runtime_error{std::format("{} must be an unsigned integer.", name)};
            return value;
        }

        [[nodiscard]] std::vector<std::string> parse_frame_sets(const std::string& text) {
            if (text.empty()) throw std::runtime_error{"frame_sets must not be empty."};
            std::vector<std::string> frame_sets;
            std::set<std::string> seen;
            std::size_t offset{};
            while (offset <= text.size()) {
                const std::size_t comma = text.find(',', offset);
                const std::string frame_set = comma == std::string::npos ? text.substr(offset) : text.substr(offset, comma - offset);
                if (frame_set != "train" && frame_set != "test") throw std::runtime_error{std::format("frame_sets contains unknown ScalarReal frame set '{}'.", frame_set)};
                if (!seen.insert(frame_set).second) throw std::runtime_error{std::format("frame_sets contains duplicate frame set '{}'.", frame_set)};
                frame_sets.push_back(frame_set);
                if (comma == std::string::npos) break;
                offset = comma + 1u;
                if (offset == text.size()) throw std::runtime_error{"frame_sets contains a trailing comma."};
            }
            return frame_sets;
        }

        [[nodiscard]] DatasetOptions parse_dataset_options(const std::span<const plugin::Option> options) {
            DatasetOptions parsed;
            std::optional<std::string> dataset_option;
            std::set<std::string> seen_options;
            for (const plugin::Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error{std::format("scene plugin open option '{}' is duplicated.", option.key)};
                if (option.key == "dataset") dataset_option = option.value;
                else if (option.key == "frame_sets") parsed.frame_sets = parse_frame_sets(option.value);
                else if (option.key == "scene_scale") parsed.scene_scale = parse_float(option.value, "scene_scale");
                else if (option.key == "view_stride") parsed.view_stride = parse_u64(option.value, "view_stride");
                else if (option.key == "max_views") parsed.max_views = parse_u64(option.value, "max_views");
                else throw std::runtime_error{std::format("unknown HyFluid project open option '{}'.", option.key)};
            }
            if (!dataset_option.has_value() || dataset_option->empty()) throw std::runtime_error{"dataset option is required."};
            parsed.dataset_path = std::filesystem::absolute(std::filesystem::path{*dataset_option}).lexically_normal();
            if (!std::filesystem::is_directory(parsed.dataset_path)) throw std::runtime_error{std::format("{}: dataset option must name an existing directory.", parsed.dataset_path.string())};
            if (!std::isfinite(parsed.scene_scale) || parsed.scene_scale <= 0.0f) throw std::runtime_error{"scene_scale must be finite and positive."};
            if (parsed.view_stride == 0u) throw std::runtime_error{"view_stride must be at least 1."};
            if (parsed.view_stride > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error{"view_stride must fit in uint32."};
            if (parsed.max_views > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error{"max_views must fit in uint32."};
            return parsed;
        }

        [[nodiscard]] Vector3 subtract(const Vector3 a, const Vector3 b) {
            return Vector3{a.x - b.x, a.y - b.y, a.z - b.z};
        }

        [[nodiscard]] Vector3 multiply(const Vector3 a, const float value) {
            return Vector3{a.x * value, a.y * value, a.z * value};
        }

        [[nodiscard]] Vector3 add(const Vector3 a, const Vector3 b) {
            return Vector3{a.x + b.x, a.y + b.y, a.z + b.z};
        }

        [[nodiscard]] Vector3 cross(const Vector3 a, const Vector3 b) {
            return Vector3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }

        [[nodiscard]] float dot(const Vector3 a, const Vector3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        [[nodiscard]] Vector3 normalize(const Vector3 value, const std::string_view context) {
            const float length_squared = value.x * value.x + value.y * value.y + value.z * value.z;
            if (!std::isfinite(length_squared) || length_squared <= 1.0e-12f) throw std::runtime_error{std::format("{} vector is degenerate.", context)};
            const float inverse_length = 1.0f / std::sqrt(length_squared);
            return Vector3{value.x * inverse_length, value.y * inverse_length, value.z * inverse_length};
        }

        [[nodiscard]] std::array<float, 3u> array_from(const Vector3 value) {
            return {value.x, value.y, value.z};
        }

        [[nodiscard]] Vector3 scalar_real_normalized_to_spectra(const Vector3 value) {
            return Vector3{value.z, value.y, value.x};
        }

        [[nodiscard]] CameraBasis spectra_image_down_camera_basis(Vector3 right, Vector3 down, Vector3 forward, const std::string_view context) {
            right = normalize(right, std::format("{} right", context));
            down = normalize(down, std::format("{} down", context));
            forward = normalize(forward, std::format("{} forward", context));
            if (std::abs(dot(right, down)) > 1.0e-3f) throw std::runtime_error{std::format("{} right/down axes are not orthogonal.", context)};
            if (std::abs(dot(right, forward)) > 1.0e-3f) throw std::runtime_error{std::format("{} right/forward axes are not orthogonal.", context)};
            if (std::abs(dot(down, forward)) > 1.0e-3f) throw std::runtime_error{std::format("{} down/forward axes are not orthogonal.", context)};
            if (dot(cross(right, down), forward) <= 0.0f) throw std::runtime_error{std::format("{} basis must satisfy cross(right, down) == forward in Spectra image-down camera space.", context)};
            return CameraBasis{
                .right = right,
                .down = down,
                .forward = forward,
            };
        }

        [[nodiscard]] CameraBasis nearest_spectra_image_down_camera_basis(const Vector3 right, const Vector3 down, const Vector3 forward, const std::string_view context) {
            CameraBasis basis{
                .right = scalar_real_normalized_to_spectra(right),
                .down = scalar_real_normalized_to_spectra(down),
                .forward = scalar_real_normalized_to_spectra(forward),
            };

            // Newton iteration for the polar factor of ScalarReal's non-uniformly scaled camera matrix.
            for (std::uint32_t iteration = 0u; iteration < 8u; ++iteration) {
                const float determinant = dot(cross(basis.right, basis.down), basis.forward);
                if (!std::isfinite(determinant) || determinant <= 1.0e-8f) throw std::runtime_error{std::format("{} ScalarReal camera transform must become right-handed after SpectraYUp mapping.", context)};
                const Vector3 inverse_transpose_right = multiply(cross(basis.down, basis.forward), 1.0f / determinant);
                const Vector3 inverse_transpose_down = multiply(cross(basis.forward, basis.right), 1.0f / determinant);
                const Vector3 inverse_transpose_forward = multiply(cross(basis.right, basis.down), 1.0f / determinant);
                basis = CameraBasis{
                    .right = multiply(add(basis.right, inverse_transpose_right), 0.5f),
                    .down = multiply(add(basis.down, inverse_transpose_down), 0.5f),
                    .forward = multiply(add(basis.forward, inverse_transpose_forward), 0.5f),
                };
            }

            return spectra_image_down_camera_basis(basis.right, basis.down, basis.forward, context);
        }

        [[nodiscard]] std::string joined_frame_sets(const std::vector<std::string>& frame_sets) {
            std::string text;
            for (const std::string& frame_set : frame_sets) {
                if (!text.empty()) text += ",";
                text += frame_set;
            }
            return text;
        }

        [[nodiscard]] plugin::Camera overview_camera() {
            constexpr Vector3 overview_target{0.5f, 0.5f, 0.5f};
            constexpr Vector3 overview_eye{0.5f, 1.55f, -1.65f};
            constexpr Vector3 overview_up{0.0f, 1.0f, 0.0f};
            const Vector3 overview_forward = normalize(subtract(overview_target, overview_eye), "overview camera forward");
            const Vector3 overview_down = multiply(overview_up, -1.0f);
            const Vector3 overview_right = normalize(cross(overview_down, overview_forward), "overview camera right");
            const Vector3 overview_camera_down = cross(overview_forward, overview_right);
            const CameraBasis basis = spectra_image_down_camera_basis(overview_right, overview_camera_down, overview_forward, "overview camera");
            return plugin::Camera{
                .name                 = "Overview",
                .position             = array_from(overview_eye),
                .right                = array_from(basis.right),
                .down                 = array_from(basis.down),
                .forward              = array_from(basis.forward),
                .projection           = plugin::CameraProjection::Perspective,
                .vertical_fov_degrees = 45.0f,
                .near_plane           = 0.01f,
                .far_plane            = 20.0f,
            };
        }

        [[nodiscard]] plugin::Camera frame_camera(const dataset::scalar_real::Frame& frame, const std::string& name, const float near_plane, const float far_plane) {
            const Vector3 camera_x{frame.camera.at(0u), frame.camera.at(1u), frame.camera.at(2u)};
            const Vector3 camera_y{frame.camera.at(3u), frame.camera.at(4u), frame.camera.at(5u)};
            const Vector3 camera_z{frame.camera.at(6u), frame.camera.at(7u), frame.camera.at(8u)};
            const Vector3 origin = scalar_real_normalized_to_spectra(Vector3{frame.camera.at(9u), frame.camera.at(10u), frame.camera.at(11u)});
            const CameraBasis basis = nearest_spectra_image_down_camera_basis(camera_x, camera_y, camera_z, std::format("dataset camera '{}'", name));
            const std::uint64_t expected_bytes = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 4u;
            if (frame.width == 0u || frame.height == 0u) throw std::runtime_error{std::format("dataset camera '{}' image dimensions must be non-zero.", name)};
            if (frame.rgba.size() != expected_bytes) throw std::runtime_error{std::format("dataset camera '{}' image byte count does not match width * height * 4.", name)};
            return plugin::Camera{
                .name                 = name,
                .position             = array_from(origin),
                .right                = array_from(basis.right),
                .down                 = array_from(basis.down),
                .forward              = array_from(basis.forward),
                .projection           = plugin::CameraProjection::Pinhole,
                .vertical_fov_degrees = 45.0f,
                .image_width          = frame.width,
                .image_height         = frame.height,
                .fx                   = frame.focal_x,
                .fy                   = frame.focal_y,
                .cx                   = frame.principal_x,
                .cy                   = frame.principal_y,
                .near_plane           = near_plane,
                .far_plane            = far_plane,
                .image                = plugin::CameraImage{
                    .rgba8      = frame.rgba.data(),
                    .rgba8_size = expected_bytes,
                    .revision   = static_cast<std::uint64_t>(frame.time_index) + 1u,
                    .width      = frame.width,
                    .height     = frame.height,
                },
            };
        }
    } // namespace

    struct Project::State final {
        DatasetOptions dataset_options;
        dataset::scalar_real::Dataset dataset;
        std::vector<FrameSetRuntime> frame_sets;
        std::uint64_t pixel_bytes{};
        std::uint64_t loaded_frame_count{};
        std::uint64_t timeline_frame_count{};
        double timeline_frame_rate{};
        double latest_time_seconds{};
        bool host_timeline_playing{true};
    };

    Project::Project() = default;
    Project::Project(std::unique_ptr<State> state) : state(std::move(state)) {}
    Project::Project(Project&& other) noexcept = default;
    Project& Project::operator=(Project&& other) noexcept = default;
    Project::~Project() noexcept = default;

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition{
            .id                = "hyfluid.project",
            .title             = "HyFluid Project",
            .open_action_label = "Open Dataset",
            .sections          = {
                plugin::section(section_dataset_id, "Dataset"),
                plugin::section(section_timeline_id, "Timeline"),
                plugin::section(section_diagnostics_id, "Diagnostics"),
            },
            .open_options      = {
                plugin::directory("dataset", "Dataset").describe("ScalarReal dataset root directory.").section(section_dataset_id).required_option(),
                plugin::text("frame_sets", "Frame Sets").describe("Comma-separated ScalarReal frame sets: train,test.").section(section_dataset_id).defaulted("train"),
                plugin::float_option("scene_scale", "Scene Scale", 1.0f).describe("ScalarReal scene scale passed to the dataset loader.").section(section_dataset_id),
                plugin::unsigned_integer("view_stride", "View Stride", 1u).describe("Only every Nth view is visualized.").section(section_dataset_id),
                plugin::unsigned_integer("max_views", "Max Views", 0u).describe("0 means no view count limit.").section(section_dataset_id),
            },
        };
        return definition;
    }

    Project Project::open(plugin::OpenContext context) {
        DatasetOptions options = parse_dataset_options(std::span<const plugin::Option>{context.options});
        std::expected<dataset::scalar_real::Dataset, std::string> loaded_dataset = dataset::scalar_real::load(options.dataset_path, dataset::scalar_real::LoadRequest{
                                                                                                                                    .frame_sets  = options.frame_sets,
                                                                                                                                    .scene_scale = options.scene_scale,
                                                                                                                                });
        if (!loaded_dataset) throw std::runtime_error{loaded_dataset.error()};

        auto state = std::make_unique<State>();
        state->dataset_options = std::move(options);
        state->dataset = std::move(*loaded_dataset);

        state->frame_sets.reserve(state->dataset.frame_sets.size());
        for (std::uint32_t frame_set_index = 0u; frame_set_index < state->dataset.frame_sets.size(); ++frame_set_index) {
            const dataset::scalar_real::FrameSet& frame_set = state->dataset.frame_sets.at(frame_set_index);
            FrameSetRuntime runtime{
                .name                    = frame_set.name,
                .dataset_frame_set_index = frame_set_index,
                .view_count              = frame_set.view_count,
                .time_count              = frame_set.time_count,
            };
            if (runtime.view_count == 0u || runtime.time_count == 0u) throw std::runtime_error{std::format("ScalarReal frame set '{}' must contain a dense view-time grid.", frame_set.name)};
            runtime.frame_indices.assign(static_cast<std::size_t>(runtime.view_count) * runtime.time_count, std::numeric_limits<std::uint32_t>::max());
            for (std::uint32_t frame_index = 0u; frame_index < frame_set.frames.size(); ++frame_index) {
                const dataset::scalar_real::Frame& frame = frame_set.frames.at(frame_index);
                if (frame.view_index >= runtime.view_count || frame.time_index >= runtime.time_count) throw std::runtime_error{std::format("ScalarReal frame set '{}' contains an out-of-range view-time index.", frame_set.name)};
                const std::uint32_t grid_index = frame.view_index * runtime.time_count + frame.time_index;
                if (runtime.frame_indices.at(grid_index) != std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("ScalarReal frame set '{}' contains duplicate view-time frames.", frame_set.name)};
                runtime.frame_indices.at(grid_index) = frame_index;
                state->pixel_bytes += static_cast<std::uint64_t>(frame.rgba.size());
                ++state->loaded_frame_count;
            }
            for (const std::uint32_t frame_index : runtime.frame_indices)
                if (frame_index == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("ScalarReal frame set '{}' is missing view-time frames.", frame_set.name)};

            for (const dataset::scalar_real::Video& video : state->dataset.videos) {
                if (video.frame_set != frame_set.name) continue;
                if (runtime.frame_rate == 0u) {
                    runtime.frame_rate = video.frame_rate;
                    runtime.width = video.width;
                    runtime.height = video.height;
                }
                if (runtime.frame_rate != video.frame_rate) throw std::runtime_error{std::format("ScalarReal frame set '{}' contains videos with mixed frame rates.", frame_set.name)};
                if (runtime.width != video.width || runtime.height != video.height) throw std::runtime_error{std::format("ScalarReal frame set '{}' contains videos with mixed dimensions.", frame_set.name)};
            }
            if (runtime.frame_rate == 0u) throw std::runtime_error{std::format("ScalarReal frame set '{}' has no video metadata.", frame_set.name)};

            for (std::uint32_t view_index = 0u; view_index < runtime.view_count; view_index += static_cast<std::uint32_t>(state->dataset_options.view_stride)) {
                if (state->dataset_options.max_views != 0u && runtime.visible_views.size() >= state->dataset_options.max_views) break;
                runtime.visible_views.push_back(view_index);
            }
            if (runtime.visible_views.empty()) throw std::runtime_error{std::format("ScalarReal frame set '{}' view selection is empty.", frame_set.name)};
            if (state->timeline_frame_count == 0u) {
                state->timeline_frame_count = runtime.time_count;
                state->timeline_frame_rate = static_cast<double>(runtime.frame_rate);
            } else {
                if (state->timeline_frame_count != runtime.time_count) throw std::runtime_error{"Selected ScalarReal frame sets must have the same time count for a single Spectra indexed timeline."};
                if (state->timeline_frame_rate != static_cast<double>(runtime.frame_rate)) throw std::runtime_error{"Selected ScalarReal frame sets must have the same frame rate for a single Spectra indexed timeline."};
            }
            state->frame_sets.push_back(std::move(runtime));
        }
        if (state->timeline_frame_count == 0u || state->timeline_frame_rate <= 0.0) throw std::runtime_error{"ScalarReal indexed timeline is empty."};

        return Project{std::move(state)};
    }

    void Project::update(const plugin::UpdateInfo& update) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (!std::isfinite(update.wall_delta_seconds) || update.wall_delta_seconds < 0.0) throw std::runtime_error{"HyFluid project update wall delta time is invalid."};
        if (!std::isfinite(update.scene_delta_seconds) || update.scene_delta_seconds < 0.0) throw std::runtime_error{"HyFluid project update scene delta time is invalid."};
        if (!std::isfinite(update.time_seconds) || update.time_seconds < 0.0) throw std::runtime_error{"HyFluid project update timeline time is invalid."};
        this->state->latest_time_seconds = update.time_seconds;
        this->state->host_timeline_playing = update.timeline_playing;
    }

    std::uint64_t Project::revision() const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        return 1u;
    }

    plugin::Document Project::document() const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        return plugin::Document{
            .timeline = plugin::TimelineDescriptor{
                .kind = plugin::TimelineKind::Indexed,
                .frame_rate = this->state->timeline_frame_rate,
                .frame_count = this->state->timeline_frame_count,
            },
            .active_camera_name = "Overview",
            .cameras = {overview_camera()},
        };
    }

    plugin::Frame Project::frame(const plugin::FrameInfo& frame) const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (!std::isfinite(frame.delta_seconds) || frame.delta_seconds < 0.0) throw std::runtime_error{"HyFluid project frame delta time is invalid."};
        if (!std::isfinite(frame.time_seconds) || frame.time_seconds < 0.0) throw std::runtime_error{"HyFluid project frame time is invalid."};
        if (frame.frame_index >= this->state->timeline_frame_count) throw std::runtime_error{"HyFluid project requested frame outside indexed timeline."};
        const std::uint32_t time_index = static_cast<std::uint32_t>(frame.frame_index);

        plugin::Frame output;
        for (const FrameSetRuntime& runtime : this->state->frame_sets) {
            const dataset::scalar_real::FrameSet& frame_set = this->state->dataset.frame_sets.at(runtime.dataset_frame_set_index);
            for (const std::uint32_t view_index : runtime.visible_views) {
                const std::uint32_t frame_index = runtime.frame_indices.at(view_index * runtime.time_count + time_index);
                const dataset::scalar_real::Frame& scalar_frame = frame_set.frames.at(frame_index);
                output.cameras.push_back(frame_camera(scalar_frame, std::format("{} view {:04}", runtime.name, view_index), this->state->dataset.near, this->state->dataset.far));
            }
        }
        if (output.cameras.empty()) throw std::runtime_error{"HyFluid project frame produced no dataset cameras."};
        return output;
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};

        controls.phase(this->state->host_timeline_playing ? "Playing" : "Paused")
            .headline("ScalarReal dataset visualization")
            .message(this->state->host_timeline_playing ? "Timeline is advancing through ScalarReal time slices." : "Timeline is paused.");
        controls.metric("dataset", "Dataset", this->state->dataset_options.dataset_path.filename().string()).section(section_dataset_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        controls.metric("frame_sets", "Frame Sets", joined_frame_sets(this->state->dataset_options.frame_sets)).section(section_dataset_id);
        controls.metric("frame_set_count", "Frame Set Count", static_cast<std::uint64_t>(this->state->dataset.frame_sets.size())).section(section_dataset_id);
        controls.metric("videos", "Videos", static_cast<std::uint64_t>(this->state->dataset.videos.size())).section(section_dataset_id);
        controls.metric("frames", "Frames", this->state->loaded_frame_count).section(section_dataset_id);
        controls.metric("pixel_storage_mib", "Pixel Storage MiB", static_cast<double>(this->state->pixel_bytes) / 1048576.0).section(section_diagnostics_id);
        controls.metric("scene_scale", "Scene Scale", std::format("{:.6f}", this->state->dataset.scene_scale)).section(section_diagnostics_id);
        controls.metric("near_far", "Near/Far", std::format("{:.6f} / {:.6f}", this->state->dataset.near, this->state->dataset.far)).section(section_diagnostics_id);
        controls.metric("time", "Time", std::format("{:.3f}s", this->state->latest_time_seconds)).section(section_timeline_id).display_primary().color({0.16f, 0.86f, 0.55f, 1.0f});
        for (const FrameSetRuntime& frame_set : this->state->frame_sets) {
            controls.metric(std::format("frame_set_{}", frame_set.name), frame_set.name, std::format("{} views x {} times | {}x{}", frame_set.view_count, frame_set.time_count, frame_set.width, frame_set.height)).section(section_timeline_id);
        }
    }
} // namespace hyfluid::project

extern "C" SPECTRA_SCENE_EXPORT auto spectra_scene_plugin_v13(void) -> decltype(hyfluid::plugin::export_plugin<hyfluid::project::Project>()) {
    return hyfluid::plugin::export_plugin<hyfluid::project::Project>();
}
