export module hyfluid.plugin;

import std;

export namespace hyfluid::plugin {
    enum class OptionKind : std::uint32_t {
        Text            = 0u,
        DirectoryPath   = 1u,
        Float           = 5u,
        UnsignedInteger = 6u,
    };

    struct ControlSection final {
        std::string id;
        std::string label;
    };

    struct OptionSchema final {
        std::string key;
        std::string label;
        std::string description;
        OptionKind kind{OptionKind::Text};
        bool required{};
        std::string default_value;
        std::string section_id;

        OptionSchema& describe(std::string value) {
            this->description = std::move(value);
            return *this;
        }

        OptionSchema& section(std::string value) {
            this->section_id = std::move(value);
            return *this;
        }

        OptionSchema& required_option() {
            this->required = true;
            return *this;
        }

        OptionSchema& defaulted(std::string value) {
            this->default_value = std::move(value);
            return *this;
        }
    };

    struct Option final {
        std::string key;
        std::string value;
    };

    struct OpenContext final {
        std::vector<Option> options;
    };

    struct UpdateInfo final {
        double wall_delta_seconds{};
        double scene_delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
        bool timeline_playing{};
    };

    struct FrameInfo final {
        double delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
    };

    enum class CameraProjection : std::uint32_t {
        Perspective = 0u,
        Pinhole     = 1u,
    };

    struct CameraImage final {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
    };

    struct Camera final {
        std::string name;
        std::array<float, 3u> position{};
        std::array<float, 3u> right{1.0f, 0.0f, 0.0f};
        std::array<float, 3u> down{0.0f, 1.0f, 0.0f};
        std::array<float, 3u> forward{0.0f, 0.0f, 1.0f};
        CameraProjection projection{CameraProjection::Perspective};
        float vertical_fov_degrees{};
        std::uint32_t image_width{};
        std::uint32_t image_height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        float near_plane{};
        float far_plane{};
        std::optional<CameraImage> image;
    };

    enum class TimelineKind : std::uint32_t {
        Static  = 0u,
        Live    = 1u,
        Indexed = 2u,
    };

    struct TimelineDescriptor final {
        TimelineKind kind{TimelineKind::Static};
        double frame_rate{};
        std::uint64_t frame_count{};
    };

    struct Document final {
        TimelineDescriptor timeline;
        std::string active_camera_name;
        std::vector<Camera> cameras;
    };

    struct Frame final {
        std::vector<Camera> cameras;
    };

    inline constexpr std::uint32_t ControlMetricDisplayPrimary = 1u << 0u;

    struct ControlMetric final {
        std::string key;
        std::string label;
        std::string value;
        std::string section_id;
        std::uint32_t display_flags{};
        bool has_color{};
        std::array<float, 4u> color{1.0f, 1.0f, 1.0f, 1.0f};
    };

    struct ControlState final {
        std::string phase;
        std::string headline;
        std::string message;
        std::vector<ControlMetric> metrics;
    };

    class ControlBuilder final {
    public:
        class MetricHandle final {
        public:
            explicit MetricHandle(ControlMetric& metric) : metric(&metric) {}

            MetricHandle& section(std::string value) {
                this->metric->section_id = std::move(value);
                return *this;
            }

            MetricHandle& display_primary() {
                this->metric->display_flags |= ControlMetricDisplayPrimary;
                return *this;
            }

            MetricHandle& color(std::array<float, 4u> value) {
                this->metric->color = value;
                this->metric->has_color = true;
                return *this;
            }

        private:
            ControlMetric* metric{};
        };

        ControlBuilder& phase(std::string value) {
            this->value.phase = std::move(value);
            return *this;
        }

        ControlBuilder& headline(std::string value) {
            this->value.headline = std::move(value);
            return *this;
        }

        ControlBuilder& message(std::string value) {
            this->value.message = std::move(value);
            return *this;
        }

        MetricHandle metric(std::string key, std::string label, std::string value) {
            this->value.metrics.push_back(ControlMetric{.key = std::move(key), .label = std::move(label), .value = std::move(value)});
            return MetricHandle{this->value.metrics.back()};
        }

        MetricHandle metric(std::string key, std::string label, const std::uint64_t value) {
            return this->metric(std::move(key), std::move(label), std::to_string(value));
        }

        MetricHandle metric(std::string key, std::string label, const double value) {
            return this->metric(std::move(key), std::move(label), std::format("{:.6g}", value));
        }

        [[nodiscard]] const ControlState& state() const {
            return this->value;
        }

    private:
        ControlState value{};
    };

    [[nodiscard]] inline ControlSection section(std::string id, std::string label) {
        return ControlSection{.id = std::move(id), .label = std::move(label)};
    }

    [[nodiscard]] inline OptionSchema text(std::string key, std::string label) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Text};
    }

    [[nodiscard]] inline OptionSchema directory(std::string key, std::string label) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::DirectoryPath};
    }

    [[nodiscard]] inline OptionSchema float_option(std::string key, std::string label, const float default_value) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Float, .default_value = std::format("{}", default_value)};
    }

    [[nodiscard]] inline OptionSchema unsigned_integer(std::string key, std::string label, const std::uint64_t default_value) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::UnsignedInteger, .default_value = std::to_string(default_value)};
    }

    template <typename Project>
    struct PluginDefinition final {
        std::string id;
        std::string title;
        std::string open_action_label;
        std::vector<ControlSection> sections;
        std::vector<OptionSchema> open_options;
    };

    struct TypeErasedPluginDefinition final {
        std::string id;
        std::string title;
        std::string open_action_label;
        std::vector<ControlSection> sections;
        std::vector<OptionSchema> open_options;
        void* (*open)(const OpenContext&) = nullptr;
        void (*destroy)(void*) = nullptr;
        void (*update)(void*, const UpdateInfo&) = nullptr;
        std::uint64_t (*revision)(const void*) = nullptr;
        Document (*document)(const void*) = nullptr;
        Frame (*frame)(const void*, const FrameInfo&) = nullptr;
        void (*write_controls)(const void*, ControlBuilder&) = nullptr;
    };

    template <typename Project>
    [[nodiscard]] TypeErasedPluginDefinition erase_plugin_definition(const PluginDefinition<Project>& definition) {
        return TypeErasedPluginDefinition{
            .id = definition.id,
            .title = definition.title,
            .open_action_label = definition.open_action_label,
            .sections = definition.sections,
            .open_options = definition.open_options,
            .open = [](const OpenContext& context) -> void* { return new Project(Project::open(context)); },
            .destroy = [](void* project) { delete static_cast<Project*>(project); },
            .update = [](void* project, const UpdateInfo& update) { static_cast<Project*>(project)->update(update); },
            .revision = [](const void* project) -> std::uint64_t { return static_cast<const Project*>(project)->revision(); },
            .document = [](const void* project) -> Document { return static_cast<const Project*>(project)->document(); },
            .frame = [](const void* project, const FrameInfo& frame) -> Frame { return static_cast<const Project*>(project)->frame(frame); },
            .write_controls = [](const void* project, ControlBuilder& builder) { static_cast<const Project*>(project)->write_controls(builder); },
        };
    }

    constexpr std::uint32_t plugin_abi_version = 13u;
    typedef void SpectraSceneInstance;
    typedef std::uint32_t SpectraSceneResult;
    constexpr std::uint32_t SPECTRA_SCENE_RESULT_OK = 0u;
    constexpr std::uint32_t SPECTRA_SCENE_RESULT_ERROR = 1u;
    constexpr std::uint32_t SPECTRA_SCENE_TIMELINE_STATIC = 0u;
    constexpr std::uint32_t SPECTRA_SCENE_TIMELINE_LIVE = 1u;
    constexpr std::uint32_t SPECTRA_SCENE_TIMELINE_INDEXED = 2u;
    constexpr std::uint32_t SPECTRA_SCENE_OPTION_PRESENTATION_DEFAULT = 0u;

    struct SpectraSceneOption {
        const char* key{};
        const char* value{};
    };

    struct SpectraSceneOptionSpan {
        const SpectraSceneOption* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlOptionChoiceSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlSection {
        const char* id{};
        const char* label{};
    };

    struct SpectraSceneControlSectionSpan {
        const SpectraSceneControlSection* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlOptionSchema {
        const char* key{};
        const char* label{};
        const char* description{};
        std::uint32_t kind{};
        std::uint32_t required{};
        const char* default_value{};
        const char* section_id{};
        SpectraSceneControlOptionChoiceSpan choices{};
        std::uint32_t presentation{};
        std::uint32_t has_numeric_range{};
        float numeric_min{};
        float numeric_max{};
        float numeric_step{};
    };

    struct SpectraSceneControlOptionSchemaSpan {
        const SpectraSceneControlOptionSchema* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlActionSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlMetric {
        const char* key{};
        const char* label{};
        const char* value{};
        const char* section_id{};
        std::uint32_t display_flags{};
        std::uint32_t has_color{};
        float color[4]{};
    };

    struct SpectraSceneControlMetricSpan {
        const SpectraSceneControlMetric* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlActionStateSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlStateView {
        std::uint64_t struct_size{};
        const char* phase{};
        const char* headline{};
        const char* detail{};
        SpectraSceneControlMetricSpan metrics{};
        SpectraSceneControlActionStateSpan action_states{};
    };

    struct SpectraSceneUpdateInfo {
        std::uint64_t struct_size{};
        double wall_delta_seconds{};
        double scene_delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
        std::uint32_t timeline_playing{};
    };

    struct SpectraSceneHostServices;

    struct SpectraSceneOpenInfo {
        std::uint64_t struct_size{};
        const char* plugin_path{};
        SpectraSceneOptionSpan options{};
        const SpectraSceneHostServices* host_services{};
    };

    struct SpectraSceneMaterialSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneLightSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneCameraImage {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
    };

    struct SpectraSceneCamera {
        const char* name{};
        float position[3]{};
        float right[3]{};
        float down[3]{};
        float forward[3]{};
        std::uint32_t projection{};
        float vertical_fov_degrees{};
        std::uint32_t image_width{};
        std::uint32_t image_height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        float near_plane{};
        float far_plane{};
        std::uint32_t has_image{};
        SpectraSceneCameraImage image{};
    };

    struct SpectraSceneCameraSpan {
        const SpectraSceneCamera* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneMeshSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneSphereSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraScenePointCloudSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneVolumeSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneViewportSegmentSetSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneViewportVoxelGridSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneItems {
        SpectraSceneMaterialSpan materials{};
        SpectraSceneLightSpan lights{};
        SpectraSceneCameraSpan cameras{};
        SpectraSceneMeshSpan meshes{};
        SpectraSceneSphereSpan spheres{};
        SpectraScenePointCloudSpan point_clouds{};
        SpectraSceneVolumeSpan volumes{};
        SpectraSceneViewportSegmentSetSpan viewport_segment_sets{};
        SpectraSceneViewportVoxelGridSpan viewport_voxel_grids{};
    };

    struct SpectraSceneTimeline {
        std::uint32_t kind{};
        double frame_rate{};
        std::uint64_t frame_count{};
    };

    struct SpectraSceneDocumentView {
        std::uint64_t struct_size{};
        SpectraSceneTimeline timeline{};
        const char* active_camera_name{};
        SpectraSceneItems items{};
    };

    struct SpectraSceneFrameInfo {
        double delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
    };

    struct SpectraSceneFrameView {
        std::uint64_t struct_size{};
        SpectraSceneItems items{};
    };

    typedef SpectraSceneResult (*SpectraSceneCreateFn)(const SpectraSceneOpenInfo*, SpectraSceneInstance**);
    typedef void (*SpectraSceneDestroyFn)(SpectraSceneInstance*);
    typedef SpectraSceneResult (*SpectraSceneUpdateFn)(SpectraSceneInstance*, const SpectraSceneUpdateInfo*);
    typedef SpectraSceneResult (*SpectraSceneDocumentFn)(SpectraSceneInstance*, SpectraSceneDocumentView*);
    typedef SpectraSceneResult (*SpectraSceneFrameFn)(SpectraSceneInstance*, SpectraSceneFrameInfo, SpectraSceneFrameView*);
    typedef SpectraSceneResult (*SpectraSceneRevisionFn)(SpectraSceneInstance*, std::uint64_t*);
    typedef SpectraSceneResult (*SpectraSceneControlActionFn)(SpectraSceneInstance*, const char*, SpectraSceneOptionSpan);
    typedef SpectraSceneResult (*SpectraSceneControlSettingUpdateFn)(SpectraSceneInstance*, const char*, const char*);
    typedef SpectraSceneResult (*SpectraSceneControlStateFn)(SpectraSceneInstance*, SpectraSceneControlStateView*);
    typedef const char* (*SpectraSceneLastErrorFn)(SpectraSceneInstance*);

    struct SpectraScenePlugin {
        std::uint32_t abi_version{};
        std::uint64_t struct_size{};
        const char* id{};
        const char* title{};
        const char* open_action_label{};
        SpectraSceneControlSectionSpan sections{};
        SpectraSceneControlOptionSchemaSpan open_options{};
        SpectraSceneControlActionSpan control_actions{};
        SpectraSceneControlOptionSchemaSpan control_settings{};
        SpectraSceneCreateFn create{};
        SpectraSceneDestroyFn destroy{};
        SpectraSceneUpdateFn update{};
        SpectraSceneDocumentFn document{};
        SpectraSceneFrameFn frame{};
        SpectraSceneRevisionFn scene_revision{};
        SpectraSceneControlActionFn control_action{};
        SpectraSceneControlSettingUpdateFn control_setting_update{};
        SpectraSceneControlStateFn control_state{};
        SpectraSceneLastErrorFn last_error{};
    };

    struct ItemAbiStorage final {
        std::vector<SpectraSceneCamera> camera_views;
    };

    struct SceneAbiStorage final {
        Document document;
        ItemAbiStorage items;
    };

    struct FrameAbiStorage final {
        Frame frame;
        ItemAbiStorage items;
    };

    struct ControlStateAbiStorage final {
        ControlState state;
        std::vector<SpectraSceneControlMetric> metric_views;
    };

    struct PluginDescriptorStorage final {
        std::vector<SpectraSceneControlSection> sections;
        std::vector<SpectraSceneControlOptionSchema> open_options;
        SpectraScenePlugin plugin;
    };

    struct PluginInstance final {
        const TypeErasedPluginDefinition* definition{};
        void* project{};
        SceneAbiStorage scene_abi;
        FrameAbiStorage frame_abi;
        ControlStateAbiStorage control_abi;
        std::string last_error;
    };

    [[nodiscard]] inline std::uint32_t option_kind_abi(const OptionKind kind) {
        return static_cast<std::uint32_t>(kind);
    }

    template <std::size_t Count>
    inline void copy_array(float (&destination)[Count], const std::array<float, Count>& source) {
        for (std::size_t index = 0u; index < Count; ++index) destination[index] = source[index];
    }

    [[nodiscard]] inline SpectraSceneCamera make_camera_view(const Camera& camera) {
        SpectraSceneCamera view{
            .name = camera.name.c_str(),
            .projection = static_cast<std::uint32_t>(camera.projection),
            .vertical_fov_degrees = camera.vertical_fov_degrees,
            .image_width = camera.image_width,
            .image_height = camera.image_height,
            .fx = camera.fx,
            .fy = camera.fy,
            .cx = camera.cx,
            .cy = camera.cy,
            .near_plane = camera.near_plane,
            .far_plane = camera.far_plane,
            .has_image = camera.image.has_value() ? 1u : 0u,
        };
        copy_array(view.position, camera.position);
        copy_array(view.right, camera.right);
        copy_array(view.down, camera.down);
        copy_array(view.forward, camera.forward);
        if (camera.image.has_value()) {
            const CameraImage& image = *camera.image;
            view.image = SpectraSceneCameraImage{.rgba8 = image.rgba8, .rgba8_size = image.rgba8_size, .revision = image.revision, .width = image.width, .height = image.height};
        }
        return view;
    }

    inline void make_item_views(ItemAbiStorage& storage, const std::vector<Camera>& cameras) {
        storage.camera_views.clear();
        storage.camera_views.reserve(cameras.size());
        for (const Camera& camera : cameras) storage.camera_views.push_back(make_camera_view(camera));
    }

    [[nodiscard]] inline SpectraSceneItems make_items_view(const ItemAbiStorage& storage) {
        return SpectraSceneItems{
            .cameras = SpectraSceneCameraSpan{.data = storage.camera_views.empty() ? nullptr : storage.camera_views.data(), .count = static_cast<std::uint64_t>(storage.camera_views.size())},
        };
    }

    [[nodiscard]] inline SpectraSceneTimeline make_timeline_view(const TimelineDescriptor& timeline) {
        switch (timeline.kind) {
        case TimelineKind::Static: return SpectraSceneTimeline{.kind = SPECTRA_SCENE_TIMELINE_STATIC, .frame_rate = timeline.frame_rate, .frame_count = timeline.frame_count};
        case TimelineKind::Live: return SpectraSceneTimeline{.kind = SPECTRA_SCENE_TIMELINE_LIVE, .frame_rate = timeline.frame_rate, .frame_count = timeline.frame_count};
        case TimelineKind::Indexed: return SpectraSceneTimeline{.kind = SPECTRA_SCENE_TIMELINE_INDEXED, .frame_rate = timeline.frame_rate, .frame_count = timeline.frame_count};
        }
        throw std::runtime_error{"plugin timeline kind is invalid"};
    }

    [[nodiscard]] inline SpectraSceneDocumentView make_document_abi_view(SceneAbiStorage& cache) {
        make_item_views(cache.items, cache.document.cameras);
        return SpectraSceneDocumentView{
            .struct_size = sizeof(SpectraSceneDocumentView),
            .timeline = make_timeline_view(cache.document.timeline),
            .active_camera_name = cache.document.active_camera_name.c_str(),
            .items = make_items_view(cache.items),
        };
    }

    [[nodiscard]] inline SpectraSceneFrameView make_frame_abi_view(FrameAbiStorage& cache) {
        make_item_views(cache.items, cache.frame.cameras);
        return SpectraSceneFrameView{
            .struct_size = sizeof(SpectraSceneFrameView),
            .items = make_items_view(cache.items),
        };
    }

    [[nodiscard]] inline SpectraSceneControlStateView make_control_state_abi_view(ControlStateAbiStorage& cache) {
        cache.metric_views.clear();
        cache.metric_views.reserve(cache.state.metrics.size());
        for (const ControlMetric& metric : cache.state.metrics) {
            SpectraSceneControlMetric view{
                .key = metric.key.c_str(),
                .label = metric.label.c_str(),
                .value = metric.value.c_str(),
                .section_id = metric.section_id.c_str(),
                .display_flags = metric.display_flags,
                .has_color = metric.has_color ? 1u : 0u,
            };
            copy_array(view.color, metric.color);
            cache.metric_views.push_back(view);
        }
        return SpectraSceneControlStateView{
            .struct_size = sizeof(SpectraSceneControlStateView),
            .phase = cache.state.phase.c_str(),
            .headline = cache.state.headline.c_str(),
            .detail = cache.state.message.c_str(),
            .metrics = SpectraSceneControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
        };
    }

    [[nodiscard]] inline PluginDescriptorStorage make_plugin_descriptor_storage(const TypeErasedPluginDefinition& definition) {
        PluginDescriptorStorage storage{};
        storage.sections.reserve(definition.sections.size());
        for (const ControlSection& section : definition.sections) storage.sections.push_back(SpectraSceneControlSection{.id = section.id.c_str(), .label = section.label.c_str()});
        storage.open_options.reserve(definition.open_options.size());
        for (const OptionSchema& option : definition.open_options) {
            storage.open_options.push_back(SpectraSceneControlOptionSchema{
                .key = option.key.c_str(),
                .label = option.label.c_str(),
                .description = option.description.c_str(),
                .kind = option_kind_abi(option.kind),
                .required = option.required ? 1u : 0u,
                .default_value = option.default_value.c_str(),
                .section_id = option.section_id.c_str(),
                .presentation = SPECTRA_SCENE_OPTION_PRESENTATION_DEFAULT,
            });
        }
        storage.plugin = SpectraScenePlugin{
            .abi_version = plugin_abi_version,
            .struct_size = sizeof(SpectraScenePlugin),
            .id = definition.id.c_str(),
            .title = definition.title.c_str(),
            .open_action_label = definition.open_action_label.c_str(),
            .sections = SpectraSceneControlSectionSpan{.data = storage.sections.empty() ? nullptr : storage.sections.data(), .count = static_cast<std::uint64_t>(storage.sections.size())},
            .open_options = SpectraSceneControlOptionSchemaSpan{.data = storage.open_options.empty() ? nullptr : storage.open_options.data(), .count = static_cast<std::uint64_t>(storage.open_options.size())},
        };
        return storage;
    }

    struct PluginExportState final {
        explicit PluginExportState(const TypeErasedPluginDefinition& definition) : definition(definition), descriptor(make_plugin_descriptor_storage(definition)) {}

        TypeErasedPluginDefinition definition;
        PluginDescriptorStorage descriptor;
        std::string global_error;

        static PluginExportState& bind(const TypeErasedPluginDefinition& definition) {
            static PluginExportState state{definition};
            active() = &state;
            return state;
        }

        static PluginExportState& current() {
            PluginExportState* state = active();
            if (state == nullptr) throw std::runtime_error{"HyFluid plugin export state is not initialized."};
            return *state;
        }

    private:
        static PluginExportState*& active() {
            static PluginExportState* state{};
            return state;
        }
    };

    [[nodiscard]] inline PluginInstance& checked_instance(SpectraSceneInstance* instance, const std::string_view context) {
        if (instance == nullptr) throw std::runtime_error{std::format("{} instance pointer is null.", context)};
        return *static_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] inline std::vector<Option> make_options(const SpectraSceneOptionSpan options) {
        if (options.count != 0u && options.data == nullptr) throw std::runtime_error{"HyFluid plugin open options pointer is null."};
        std::vector<Option> result;
        result.reserve(static_cast<std::size_t>(options.count));
        for (std::uint64_t index = 0u; index < options.count; ++index) {
            const SpectraSceneOption& option = options.data[index];
            if (option.key == nullptr) throw std::runtime_error{"HyFluid plugin open option key is null."};
            if (option.value == nullptr) throw std::runtime_error{std::format("HyFluid plugin open option '{}' value is null.", option.key)};
            result.push_back(Option{.key = option.key, .value = option.value});
        }
        return result;
    }

    [[nodiscard]] inline SpectraSceneResult scene_create(const SpectraSceneOpenInfo* open_info, SpectraSceneInstance** instance) noexcept {
        try {
            if (open_info == nullptr) throw std::runtime_error{"HyFluid plugin create open info pointer is null."};
            if (instance == nullptr) throw std::runtime_error{"HyFluid plugin create instance output pointer is null."};
            if (open_info->struct_size != sizeof(SpectraSceneOpenInfo)) throw std::runtime_error{"HyFluid plugin open info ABI size mismatch."};
            *instance = nullptr;
            PluginExportState& state = PluginExportState::current();
            state.global_error.clear();
            auto plugin_instance = std::make_unique<PluginInstance>();
            plugin_instance->definition = &state.definition;
            plugin_instance->project = state.definition.open(OpenContext{.options = make_options(open_info->options)});
            *instance = reinterpret_cast<SpectraSceneInstance*>(plugin_instance.release());
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            PluginExportState::current().global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    inline void scene_destroy(SpectraSceneInstance* instance) noexcept {
        auto plugin_instance = static_cast<PluginInstance*>(instance);
        if (plugin_instance == nullptr) return;
        if (plugin_instance->definition != nullptr && plugin_instance->project != nullptr) plugin_instance->definition->destroy(plugin_instance->project);
        delete plugin_instance;
    }

    [[nodiscard]] inline SpectraSceneResult scene_update(SpectraSceneInstance* instance, const SpectraSceneUpdateInfo* update_info) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "HyFluid plugin update");
            if (update_info == nullptr) throw std::runtime_error{"HyFluid plugin update info pointer is null."};
            if (update_info->struct_size != sizeof(SpectraSceneUpdateInfo)) throw std::runtime_error{"HyFluid plugin update info ABI size mismatch."};
            plugin_instance.last_error.clear();
            plugin_instance.definition->update(plugin_instance.project, UpdateInfo{
                .wall_delta_seconds = update_info->wall_delta_seconds,
                .scene_delta_seconds = update_info->scene_delta_seconds,
                .time_seconds = update_info->time_seconds,
                .frame_index = update_info->frame_index,
                .timeline_playing = update_info->timeline_playing != 0u,
            });
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] inline SpectraSceneResult scene_document(SpectraSceneInstance* instance, SpectraSceneDocumentView* document) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "HyFluid plugin document");
            if (document == nullptr) throw std::runtime_error{"HyFluid plugin document output pointer is null."};
            plugin_instance.last_error.clear();
            plugin_instance.scene_abi.document = plugin_instance.definition->document(plugin_instance.project);
            *document = make_document_abi_view(plugin_instance.scene_abi);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] inline SpectraSceneResult scene_frame(SpectraSceneInstance* instance, const SpectraSceneFrameInfo frame, SpectraSceneFrameView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "HyFluid plugin frame");
            if (snapshot == nullptr) throw std::runtime_error{"HyFluid plugin frame output pointer is null."};
            plugin_instance.last_error.clear();
            plugin_instance.frame_abi.frame = plugin_instance.definition->frame(plugin_instance.project, FrameInfo{.delta_seconds = frame.delta_seconds, .time_seconds = frame.time_seconds, .frame_index = frame.frame_index});
            *snapshot = make_frame_abi_view(plugin_instance.frame_abi);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] inline SpectraSceneResult scene_revision(SpectraSceneInstance* instance, std::uint64_t* revision) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "HyFluid plugin revision");
            if (revision == nullptr) throw std::runtime_error{"HyFluid plugin revision output pointer is null."};
            plugin_instance.last_error.clear();
            *revision = plugin_instance.definition->revision(plugin_instance.project);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] inline SpectraSceneResult scene_control_action(SpectraSceneInstance* instance, const char*, SpectraSceneOptionSpan) noexcept {
        if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = "HyFluid project plugin does not expose control actions.";
        return SPECTRA_SCENE_RESULT_ERROR;
    }

    [[nodiscard]] inline SpectraSceneResult scene_control_setting_update(SpectraSceneInstance* instance, const char*, const char*) noexcept {
        if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = "HyFluid project plugin does not expose control settings.";
        return SPECTRA_SCENE_RESULT_ERROR;
    }

    [[nodiscard]] inline SpectraSceneResult scene_control_state(SpectraSceneInstance* instance, SpectraSceneControlStateView* state) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "HyFluid plugin control state");
            if (state == nullptr) throw std::runtime_error{"HyFluid plugin control state output pointer is null."};
            plugin_instance.last_error.clear();
            ControlBuilder builder;
            plugin_instance.definition->write_controls(plugin_instance.project, builder);
            plugin_instance.control_abi.state = builder.state();
            *state = make_control_state_abi_view(plugin_instance.control_abi);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) static_cast<PluginInstance*>(instance)->last_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] inline const char* scene_last_error(SpectraSceneInstance* instance) noexcept {
        if (instance == nullptr) return PluginExportState::current().global_error.c_str();
        return static_cast<PluginInstance*>(instance)->last_error.c_str();
    }

    [[nodiscard]] inline const SpectraScenePlugin* export_type_erased_plugin(const TypeErasedPluginDefinition& definition) {
        PluginExportState& state = PluginExportState::bind(definition);
        state.descriptor.plugin.create = scene_create;
        state.descriptor.plugin.destroy = scene_destroy;
        state.descriptor.plugin.update = scene_update;
        state.descriptor.plugin.document = scene_document;
        state.descriptor.plugin.frame = scene_frame;
        state.descriptor.plugin.scene_revision = scene_revision;
        state.descriptor.plugin.control_action = scene_control_action;
        state.descriptor.plugin.control_setting_update = scene_control_setting_update;
        state.descriptor.plugin.control_state = scene_control_state;
        state.descriptor.plugin.last_error = scene_last_error;
        return &state.descriptor.plugin;
    }

    template <typename Project>
    [[nodiscard]] const SpectraScenePlugin* export_plugin() {
        static const TypeErasedPluginDefinition definition = erase_plugin_definition(Project::plugin());
        return export_type_erased_plugin(definition);
    }
} // namespace hyfluid::plugin
