module;

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

#if defined(_WIN32)
#define SPECTRA_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_SCENE_EXPORT __attribute__((visibility("default")))
#endif

module hyfluid.project;

import dataset.scalar_real;
import hyfluid.plugin;
import hyfluid.train;
import hyfluid.inspector;
import std;

namespace hyfluid::project {
    namespace {
        constexpr char section_dataset_id[] = "dataset";
        constexpr char section_timeline_id[] = "timeline";
        constexpr char section_sampler_id[] = "sampler";
        constexpr char section_diagnostics_id[] = "diagnostics";
        constexpr char setting_show_sampler_key[] = "show_sampler";
        constexpr char setting_show_sampler_points_key[] = "show_sampler_points";
        constexpr char setting_show_sampler_rays_key[] = "show_sampler_rays";
        constexpr char setting_sampler_point_radius_key[] = "sampler_point_radius";
        constexpr char setting_sampler_ray_width_key[] = "sampler_ray_width";
        constexpr char sampler_point_cloud_name[] = "HyFluid Sampler Samples";
        constexpr char sampler_ray_segments_name[] = "HyFluid Sampler Rays";
        constexpr char sampler_material_name[] = "HyFluid Sampler Point Material";

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

        struct DebugOptions final {
            bool show_sampler{false};
            bool show_sampler_points{true};
            bool show_sampler_rays{true};
            float sampler_point_radius{0.002f};
            float sampler_ray_width{1.5f};
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

        class ExternalGpuBuffer final {
        public:
            ExternalGpuBuffer() = default;
            ExternalGpuBuffer(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer(ExternalGpuBuffer&&) = delete;
            ExternalGpuBuffer& operator=(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer& operator=(ExternalGpuBuffer&&) = delete;
            ~ExternalGpuBuffer() noexcept;

            void ensure(std::shared_ptr<plugin::HostServices> host_services, std::uint32_t kind, std::uint64_t requested_byte_size, std::string_view debug_name, std::string_view label);
            void reset() noexcept;

            [[nodiscard]] bool has_capacity(std::uint64_t requested_byte_size) const noexcept;
            [[nodiscard]] std::uint64_t resource_id() const noexcept;

            template <typename Value>
            [[nodiscard]] Value* mapped_as() const noexcept {
                return static_cast<Value*>(this->mapped_buffer);
            }

        private:
            std::shared_ptr<plugin::HostServices> host_services{};
            plugin::GpuBufferAllocation allocation{};
            cudaExternalMemory_t external_memory{};
            void* mapped_buffer{};
            std::uint64_t byte_size{};
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
                .name = "Overview",
                .position = array_from(overview_eye),
                .right = array_from(basis.right),
                .down = array_from(basis.down),
                .forward = array_from(basis.forward),
                .projection = plugin::CameraProjection::Perspective,
                .vertical_fov_degrees = 45.0f,
                .near_plane = 0.01f,
                .far_plane = 20.0f,
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
                .name = name,
                .position = array_from(origin),
                .right = array_from(basis.right),
                .down = array_from(basis.down),
                .forward = array_from(basis.forward),
                .projection = plugin::CameraProjection::Pinhole,
                .vertical_fov_degrees = 45.0f,
                .image_width = frame.width,
                .image_height = frame.height,
                .fx = frame.focal_x,
                .fy = frame.focal_y,
                .cx = frame.principal_x,
                .cy = frame.principal_y,
                .near_plane = near_plane,
                .far_plane = far_plane,
                .image = plugin::CameraImage{
                    .rgba8 = frame.rgba.data(),
                    .rgba8_size = expected_bytes,
                    .revision = static_cast<std::uint64_t>(frame.time_index) + 1u,
                    .width = frame.width,
                    .height = frame.height,
                },
            };
        }

        [[nodiscard]] bool has_nonzero_bytes(const std::span<const std::uint8_t> bytes) {
            return std::ranges::any_of(bytes, [](const std::uint8_t value) { return value != 0u; });
        }

        void close_imported_handle(plugin::GpuBufferAllocation& allocation) noexcept {
#if defined(_WIN32)
            if (allocation.handle_kind == plugin::GpuResourceHandleKind::OpaqueWin32 && allocation.handle != 0u) static_cast<void>(CloseHandle(reinterpret_cast<HANDLE>(allocation.handle)));
#else
            if (allocation.handle_kind == plugin::GpuResourceHandleKind::OpaqueFileDescriptor && allocation.handle != 0u) static_cast<void>(close(static_cast<int>(allocation.handle)));
#endif
            allocation.handle = 0u;
        }

        void validate_cuda_device_identity(const plugin::GpuDeviceIdentity& identity) {
            int cuda_device = -1;
            if (const cudaError_t status = cudaGetDevice(&cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDevice failed: "} + cudaGetErrorString(status)};
            cudaDeviceProp device_properties{};
            if (const cudaError_t status = cudaGetDeviceProperties(&device_properties, cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDeviceProperties failed: "} + cudaGetErrorString(status)};

            const bool has_vulkan_uuid = has_nonzero_bytes(std::span<const std::uint8_t>{identity.device_uuid.data(), identity.device_uuid.size()});
            const bool has_vulkan_luid = has_nonzero_bytes(std::span<const std::uint8_t>{identity.device_luid.data(), identity.device_luid.size()});
            if (!has_vulkan_uuid && !has_vulkan_luid) throw std::runtime_error{"Spectra GPU resource device identity did not include UUID or LUID."};
            if (has_vulkan_uuid) {
                const std::uint8_t* const cuda_uuid = reinterpret_cast<const std::uint8_t*>(device_properties.uuid.bytes);
                for (std::size_t index = 0u; index < identity.device_uuid.size(); ++index)
                    if (identity.device_uuid[index] != cuda_uuid[index]) throw std::runtime_error{"Spectra Vulkan device UUID does not match the active CUDA device."};
            }
#if defined(_WIN32)
            if (has_vulkan_luid) {
                const std::uint8_t* const cuda_luid = reinterpret_cast<const std::uint8_t*>(device_properties.luid);
                for (std::size_t index = 0u; index < identity.device_luid.size(); ++index)
                    if (identity.device_luid[index] != cuda_luid[index]) throw std::runtime_error{"Spectra Vulkan device LUID does not match the active CUDA device."};
                if (identity.device_node_mask != device_properties.luidDeviceNodeMask) throw std::runtime_error{"Spectra Vulkan device node mask does not match the active CUDA device."};
            }
#endif
        }
    }

    struct Project::State final {
        DatasetOptions dataset_options;
        DebugOptions debug;
        dataset::scalar_real::Dataset dataset;
        std::unique_ptr<train::HyFluid> trainer;
        std::shared_ptr<plugin::HostServices> host_services;
        std::vector<FrameSetRuntime> frame_sets;
        std::optional<train::OptimizationStats> latest_stats;
        ExternalGpuBuffer sampler_points_buffer{};
        ExternalGpuBuffer sampler_segments_buffer{};
        std::optional<plugin::PointCloud> sampler_point_cloud{};
        std::optional<plugin::ViewportSegmentSet> sampler_ray_segments{};
        plugin::DebugAttachmentSet debug_attachments{};
        std::uint64_t pixel_bytes{};
        std::uint64_t loaded_frame_count{};
        std::uint64_t timeline_frame_count{};
        std::uint64_t scene_revision{1u};
        std::uint64_t exported_sampler_revision{};
        std::uint32_t exported_sampler_point_count{};
        std::uint32_t exported_sampler_ray_count{};
        double timeline_frame_rate{};
        bool host_update_running{};
    };

    ExternalGpuBuffer::~ExternalGpuBuffer() noexcept {
        this->reset();
    }

    void ExternalGpuBuffer::reset() noexcept {
        this->mapped_buffer = nullptr;
        if (this->external_memory != nullptr) {
            static_cast<void>(cudaDestroyExternalMemory(this->external_memory));
            this->external_memory = nullptr;
        }
        if (this->allocation.resource_id != 0u && this->host_services != nullptr) {
            try {
                this->host_services->release_gpu_buffer(this->allocation.resource_id);
            } catch (...) {
            }
        }
        this->allocation = plugin::GpuBufferAllocation{};
        this->byte_size = 0u;
        this->host_services.reset();
    }

    bool ExternalGpuBuffer::has_capacity(const std::uint64_t requested_byte_size) const noexcept {
        return this->allocation.resource_id != 0u && this->byte_size >= requested_byte_size;
    }

    std::uint64_t ExternalGpuBuffer::resource_id() const noexcept {
        return this->allocation.resource_id;
    }

    void ExternalGpuBuffer::ensure(std::shared_ptr<plugin::HostServices> next_host_services, const std::uint32_t kind, const std::uint64_t requested_byte_size, const std::string_view debug_name, const std::string_view label) {
        if (this->has_capacity(requested_byte_size)) return;
        this->reset();
        if (next_host_services == nullptr) throw std::runtime_error{std::format("Spectra host services are required for {} visualization.", label)};
        if (requested_byte_size == 0u) throw std::runtime_error{std::format("{} byte size is invalid.", label)};
        if (!next_host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
        if (!next_host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
        plugin::GpuBufferAllocation next_allocation = next_host_services->request_gpu_buffer(kind, requested_byte_size, debug_name);
        if (next_allocation.resource_id == 0u) throw std::runtime_error{std::format("Spectra returned an invalid {} resource id.", label)};
        if (next_allocation.kind != kind) throw std::runtime_error{std::format("Spectra returned an unexpected GPU buffer kind for {}.", label)};
        if (next_allocation.byte_size < requested_byte_size) {
            close_imported_handle(next_allocation);
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw std::runtime_error{std::format("Spectra returned a {} buffer smaller than requested.", label)};
        }
        if (next_allocation.handle == 0u) {
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw std::runtime_error{std::format("Spectra returned an empty {} external memory handle.", label)};
        }

        try {
            validate_cuda_device_identity(next_allocation.device_identity);
            cudaExternalMemoryHandleDesc memory_desc{};
            memory_desc.size = next_allocation.byte_size;
            switch (next_allocation.handle_kind) {
#if defined(_WIN32)
                case plugin::GpuResourceHandleKind::OpaqueWin32:
                    memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                    memory_desc.handle.win32.handle = reinterpret_cast<void*>(next_allocation.handle);
                    break;
#else
                case plugin::GpuResourceHandleKind::OpaqueFileDescriptor:
                    memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                    memory_desc.handle.fd = static_cast<int>(next_allocation.handle);
                    break;
#endif
                default:
                    throw std::runtime_error{std::format("Spectra returned an unsupported {} external memory handle kind.", label)};
            }

            cudaExternalMemory_t imported_memory{};
            const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
            close_imported_handle(next_allocation);
            if (import_status != cudaSuccess) throw std::runtime_error{std::format("cudaImportExternalMemory for {} failed: {}", label, cudaGetErrorString(import_status))};

            cudaExternalMemoryBufferDesc buffer_desc{};
            buffer_desc.size = next_allocation.byte_size;
            void* next_mapped_buffer = nullptr;
            if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&next_mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer for {} failed: {}", label, cudaGetErrorString(status))};
            }
            if (next_mapped_buffer == nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer returned null for {}.", label)};
            }
            this->host_services = std::move(next_host_services);
            this->allocation = next_allocation;
            this->external_memory = imported_memory;
            this->mapped_buffer = next_mapped_buffer;
            this->byte_size = requested_byte_size;
        } catch (...) {
            if (next_allocation.handle != 0u) close_imported_handle(next_allocation);
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw;
        }
    }

    namespace {
        void reset_sampler_visualization(Project::State& state, const bool release_buffers) noexcept {
            state.sampler_point_cloud.reset();
            state.sampler_ray_segments.reset();
            state.exported_sampler_revision = 0u;
            state.exported_sampler_point_count = 0u;
            state.exported_sampler_ray_count = 0u;
            state.debug_attachments.viewport_segment_sets.clear();
            if (release_buffers) {
                state.sampler_points_buffer.reset();
                state.sampler_segments_buffer.reset();
            }
        }

        void publish_sampler_visualization_if_ready(Project::State& state, const std::uint64_t timeline_frame_index) {
            if (!state.debug.show_sampler || state.trainer == nullptr || (!state.debug.show_sampler_points && !state.debug.show_sampler_rays)) {
                reset_sampler_visualization(state, true);
                return;
            }

            const inspector::Inspector inspector{*state.trainer};
            const inspector::SamplerBatchDeviceView view = inspector.sampler_batch_device_view();
            if (!view.initialized) {
                reset_sampler_visualization(state, true);
                return;
            }
            if (view.sample_count == 0u || view.ray_count == 0u) throw std::runtime_error{"HyFluid sampler visualization last batch is empty."};
            if (view.sample_count > std::numeric_limits<std::uint64_t>::max() / inspector::SamplerPointInstanceBytes) throw std::runtime_error{"HyFluid sampler point visualization byte size overflows uint64."};
            if (view.ray_count > std::numeric_limits<std::uint64_t>::max() / inspector::SamplerSegmentInstanceBytes) throw std::runtime_error{"HyFluid sampler ray visualization byte size overflows uint64."};

            const std::uint64_t point_byte_size = static_cast<std::uint64_t>(view.sample_count) * inspector::SamplerPointInstanceBytes;
            const std::uint64_t segment_byte_size = static_cast<std::uint64_t>(view.ray_count) * inspector::SamplerSegmentInstanceBytes;
            if (state.timeline_frame_count == 0u || state.timeline_frame_count > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"HyFluid sampler visualization timeline frame count is invalid."};
            if (timeline_frame_index >= state.timeline_frame_count) throw std::runtime_error{"HyFluid sampler visualization frame index is outside timeline."};
            std::byte* point_instances = nullptr;
            std::byte* segment_instances = nullptr;
            std::uint64_t requested_point_bytes{};
            std::uint64_t requested_segment_bytes{};
            const bool needs_point_cloud = state.debug.show_sampler_points;

            if (needs_point_cloud) {
                state.sampler_points_buffer.ensure(state.host_services, plugin::GpuBufferKindPointCloud, point_byte_size, "hyfluid sampler samples", "sampler point cloud");
                point_instances = state.sampler_points_buffer.mapped_as<std::byte>();
                if (point_instances == nullptr) throw std::runtime_error{"HyFluid sampler point cloud external buffer was not mapped."};
                requested_point_bytes = point_byte_size;
            } else {
                state.sampler_points_buffer.reset();
                state.sampler_point_cloud.reset();
            }

            if (state.debug.show_sampler_rays) {
                state.sampler_segments_buffer.ensure(state.host_services, plugin::GpuBufferKindViewportSegmentSet, segment_byte_size, "hyfluid sampler rays", "sampler ray segments");
                segment_instances = state.sampler_segments_buffer.mapped_as<std::byte>();
                if (segment_instances == nullptr) throw std::runtime_error{"HyFluid sampler ray segment external buffer was not mapped."};
                requested_segment_bytes = segment_byte_size;
            } else {
                state.sampler_segments_buffer.reset();
                state.sampler_ray_segments.reset();
            }

            const std::expected<inspector::SamplerVisualizationStats, std::string> stats = inspector.write_sampler_visualization(inspector::SamplerVisualizationRequest{
                .point_instances = point_instances,
                .point_byte_size = requested_point_bytes,
                .segment_instances = segment_instances,
                .segment_byte_size = requested_segment_bytes,
                .time_count = static_cast<std::uint32_t>(state.timeline_frame_count),
                .time_index = static_cast<std::uint32_t>(timeline_frame_index),
                .point_radius = state.debug.sampler_point_radius,
                .ray_width = state.debug.sampler_ray_width,
                .width_mode = static_cast<std::uint32_t>(plugin::ViewportSegmentWidthMode::Screen),
            });
            if (!stats) throw std::runtime_error(stats.error());

            state.exported_sampler_revision = stats->revision;
            state.exported_sampler_point_count = needs_point_cloud ? stats->point_count : 0u;
            state.exported_sampler_ray_count = state.debug.show_sampler_rays ? stats->ray_count : 0u;
            state.sampler_point_cloud.reset();
            state.sampler_ray_segments.reset();
            state.debug_attachments.viewport_segment_sets.clear();
            if (needs_point_cloud && stats->point_count != 0u) {
                state.sampler_point_cloud = plugin::PointCloud{
                    .name = sampler_point_cloud_name,
                    .source_kind = plugin::PointCloudSourceKind::ExternalGpuBuffer,
                    .point_count = stats->point_count,
                    .buffer_id = state.sampler_points_buffer.resource_id(),
                    .source_byte_size = stats->point_byte_size,
                    .revision = stats->revision,
                    .material_name = sampler_material_name,
                    .transform = {},
                    .bounds = plugin::Bounds{
                        .minimum = {0.0f, 0.0f, 0.0f},
                        .maximum = {1.0f, 1.0f, 1.0f},
                    },
                };
            }
            if (state.debug.show_sampler_rays && stats->ray_count != 0u) {
                state.sampler_ray_segments = plugin::ViewportSegmentSet{
                    .name = sampler_ray_segments_name,
                    .owner = state.sampler_point_cloud.has_value() ? plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::PointCloud, .name = sampler_point_cloud_name} : plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::Camera, .name = "Overview"},
                    .source_kind = plugin::ViewportSegmentSourceKind::ExternalGpuBuffer,
                    .segment_count = stats->ray_count,
                    .buffer_id = state.sampler_segments_buffer.resource_id(),
                    .source_byte_size = stats->segment_byte_size,
                    .revision = stats->revision,
                    .width = state.debug.sampler_ray_width,
                    .width_mode = plugin::ViewportSegmentWidthMode::Screen,
                    .depth_mode = plugin::ViewportSegmentDepthMode::DepthTested,
                };
            }

            if (state.sampler_ray_segments.has_value()) state.debug_attachments.viewport_segment_sets.push_back(*state.sampler_ray_segments);
        }
    }

    Project::Project(std::unique_ptr<State> state) : state(std::move(state)) {}
    Project::Project(Project&& other) noexcept = default;
    Project& Project::operator=(Project&& other) noexcept = default;
    Project::~Project() noexcept = default;

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition{
            .id = "hyfluid.project",
            .title = "HyFluid Project",
            .open_action_label = "Open Dataset",
            .sections = {
                plugin::section(section_dataset_id, "Dataset"),
                plugin::section(section_timeline_id, "Timeline"),
                plugin::section(section_sampler_id, "Sampler"),
                plugin::section(section_diagnostics_id, "Diagnostics"),
            },
            .open_options = {
                plugin::directory("dataset", "Dataset").describe("ScalarReal dataset root directory.").section(section_dataset_id).defaulted(R"(C:\Users\xayah\Desktop\hyfluid-dev\data\ScalarReal)").required(),
                plugin::text("frame_sets", "Frame Sets").describe("Comma-separated ScalarReal frame sets: train,test.").section(section_dataset_id).defaulted("train"),
                plugin::float_option("scene_scale", "Scene Scale", 1.0f).describe("ScalarReal scene scale passed to the dataset loader.").section(section_dataset_id),
                plugin::unsigned_integer("view_stride", "View Stride", 1u).describe("Only every Nth view is visualized.").section(section_dataset_id),
                plugin::unsigned_integer("max_views", "Max Views", 0u).describe("0 means no view count limit.").section(section_dataset_id),
            },
            .settings = {
                plugin::toggle(setting_show_sampler_key, "Show Sampler", false, &Project::set_show_sampler).section(section_sampler_id),
                plugin::toggle(setting_show_sampler_points_key, "Show Sampler Points", true, &Project::set_show_sampler_points).section(section_sampler_id),
                plugin::toggle(setting_show_sampler_rays_key, "Show Sampler Rays", true, &Project::set_show_sampler_rays).section(section_sampler_id),
                plugin::float_value(setting_sampler_point_radius_key, "Point Radius", 0.002f, &Project::set_sampler_point_radius).section(section_sampler_id).slider(0.0001f, 0.01f, 0.0001f),
                plugin::float_value(setting_sampler_ray_width_key, "Ray Width", 1.5f, &Project::set_sampler_ray_width).section(section_sampler_id).slider(0.25f, 6.0f, 0.25f),
            },
        };
        return definition;
    }

    Project Project::open(plugin::OpenContext context) {
        DatasetOptions options = parse_dataset_options(std::span<const plugin::Option>{context.options});
        std::expected<dataset::scalar_real::Dataset, std::string> loaded_dataset = dataset::scalar_real::load(options.dataset_path, dataset::scalar_real::LoadRequest{
                                                                                                                                    .frame_sets = options.frame_sets,
                                                                                                                                    .scene_scale = options.scene_scale,
                                                                                                                                });
        if (!loaded_dataset) throw std::runtime_error{loaded_dataset.error()};

        auto state = std::make_unique<State>();
        state->dataset_options = std::move(options);
        state->dataset = std::move(*loaded_dataset);
        state->host_services = std::move(context.host_services);
        state->trainer = std::make_unique<train::HyFluid>(state->dataset);

        state->frame_sets.reserve(state->dataset.frame_sets.size());
        for (std::uint32_t frame_set_index = 0u; frame_set_index < state->dataset.frame_sets.size(); ++frame_set_index) {
            const dataset::scalar_real::FrameSet& frame_set = state->dataset.frame_sets.at(frame_set_index);
            FrameSetRuntime runtime{
                .name = frame_set.name,
                .dataset_frame_set_index = frame_set_index,
                .view_count = frame_set.view_count,
                .time_count = frame_set.time_count,
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
        if (!std::isfinite(update.update_delta_seconds) || update.update_delta_seconds < 0.0) throw std::runtime_error{"HyFluid project update delta time is invalid."};
        if (!std::isfinite(update.timeline_time_seconds) || update.timeline_time_seconds < 0.0) throw std::runtime_error{"HyFluid project timeline time is invalid."};
        if (update.timeline_frame_index >= this->state->timeline_frame_count) throw std::runtime_error{"HyFluid project update frame index is outside indexed timeline."};
        this->state->host_update_running = update.update_running;
        if (update.update_delta_seconds == 0.0) return;
        bool scene_changed = false;
        const std::expected<train::OptimizationStats, std::string> stats = this->state->trainer->optimize(train::OptimizationRequest{
            .frame_set = this->state->frame_sets.front().name,
            .iterations = 1,
        });
        if (!stats) throw std::runtime_error{stats.error()};
        this->state->latest_stats = *stats;
        scene_changed = true;
        const std::uint64_t previous_sampler_revision = this->state->exported_sampler_revision;
        publish_sampler_visualization_if_ready(*this->state, update.timeline_frame_index);
        if (this->state->exported_sampler_revision != previous_sampler_revision) scene_changed = true;
        if (scene_changed) ++this->state->scene_revision;
    }

    std::uint64_t Project::revision() const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        return this->state->scene_revision;
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        std::vector<plugin::Material> materials{};
        if (this->state->debug.show_sampler && this->state->debug.show_sampler_points) {
            materials.push_back(plugin::Material{
                .name = sampler_material_name,
                .model = "point_sprite",
                .alpha_mode = "blend",
                .base_color = {1.0f, 1.0f, 1.0f, 1.0f},
            });
        }
        scene.set_document(plugin::Document{
            .timeline = plugin::TimelineDescriptor{
                .kind = plugin::TimelineKind::Indexed,
                .frame_rate = this->state->timeline_frame_rate,
                .frame_count = this->state->timeline_frame_count,
            },
            .update = plugin::UpdateDescriptor{
                .enabled = true,
                .initial_running = false,
                .step_delta_seconds = 1.0 / 60.0,
            },
            .active_camera_name = "Overview",
            .materials = std::move(materials),
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo frame) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (!std::isfinite(frame.delta_seconds) || frame.delta_seconds < 0.0) throw std::runtime_error{"HyFluid project frame delta time is invalid."};
        if (!std::isfinite(frame.time_seconds) || frame.time_seconds < 0.0) throw std::runtime_error{"HyFluid project frame time is invalid."};
        if (frame.frame_index >= this->state->timeline_frame_count) throw std::runtime_error{"HyFluid project frame index is outside indexed timeline."};
        publish_sampler_visualization_if_ready(*this->state, frame.frame_index);
        const std::uint32_t time_index = static_cast<std::uint32_t>(frame.frame_index);
        plugin::Document document{
            .cameras = {overview_camera()},
            .debug_attachments = this->state->debug_attachments,
        };

        for (const FrameSetRuntime& runtime : this->state->frame_sets) {
            const dataset::scalar_real::FrameSet& frame_set = this->state->dataset.frame_sets.at(runtime.dataset_frame_set_index);
            for (const std::uint32_t view_index : runtime.visible_views) {
                const std::uint32_t frame_index = runtime.frame_indices.at(view_index * runtime.time_count + time_index);
                const dataset::scalar_real::Frame& scalar_frame = frame_set.frames.at(frame_index);
                document.cameras.push_back(frame_camera(scalar_frame, std::format("{} view {:04}", runtime.name, view_index), this->state->dataset.near, this->state->dataset.far));
            }
        }
        if (this->state->sampler_point_cloud.has_value()) {
            document.point_clouds.push_back(*this->state->sampler_point_cloud);
        }
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};

        controls.phase(this->state->host_update_running ? "Running" : "Paused")
            .headline(this->state->latest_stats.has_value() ? "Sampler running" : "ScalarReal dataset loaded")
            .message(this->state->latest_stats.has_value() ? "HyFluid sampler-only batches advance on Spectra update ticks." : "Update clock is paused before the first sampler batch.");
        controls.metric("dataset", "Dataset", this->state->dataset_options.dataset_path.filename().string()).section(section_dataset_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        controls.metric("frame_sets", "Frame Sets", joined_frame_sets(this->state->dataset_options.frame_sets)).section(section_dataset_id);
        controls.metric("frame_set_count", "Frame Set Count", static_cast<std::uint64_t>(this->state->dataset.frame_sets.size())).section(section_dataset_id);
        controls.metric("videos", "Videos", static_cast<std::uint64_t>(this->state->dataset.videos.size())).section(section_dataset_id);
        controls.metric("frames", "Frames", this->state->loaded_frame_count).section(section_dataset_id);
        controls.metric("pixel_storage_mib", "Pixel Storage MiB", static_cast<double>(this->state->pixel_bytes) / 1048576.0).section(section_diagnostics_id);
        controls.metric("scene_scale", "Scene Scale", std::format("{:.6f}", this->state->dataset.scene_scale)).section(section_diagnostics_id);
        controls.metric("near_far", "Near/Far", std::format("{:.6f} / {:.6f}", this->state->dataset.near, this->state->dataset.far)).section(section_diagnostics_id);
        for (const FrameSetRuntime& frame_set : this->state->frame_sets) {
            controls.metric(std::format("frame_set_{}", frame_set.name), frame_set.name, std::format("{} views x {} times | {}x{}", frame_set.view_count, frame_set.time_count, frame_set.width, frame_set.height)).section(section_timeline_id);
        }

        controls.metric("sampler_points", "Points", std::format("{} ({})", this->state->sampler_point_cloud.has_value() ? "visible" : "hidden", this->state->exported_sampler_point_count)).section(section_sampler_id);
        controls.metric("sampler_rays", "Rays", std::format("{} ({})", this->state->sampler_ray_segments.has_value() ? "visible" : "hidden", this->state->exported_sampler_ray_count)).section(section_sampler_id);
        if (this->state->latest_stats.has_value()) {
            const train::OptimizationStats& stats = *this->state->latest_stats;
            controls.metric("step", "Step", stats.step).section(section_sampler_id).display_primary().color({0.16f, 0.86f, 0.55f, 1.0f});
            controls.metric("rays", "Rays", std::format("{}/{}", stats.ray_count, stats.rays_per_batch)).section(section_sampler_id);
            controls.metric("samples", "Samples", stats.sample_count).section(section_sampler_id);
            controls.metric("sample_eff", "Sample Eff", std::format("{:.2f}%", stats.sample_efficiency_ratio * 100.0f)).section(section_sampler_id);
            controls.metric("occupancy", "Occupancy", std::format("{:.2f}%", stats.occupancy_grid_ratio * 100.0f)).section(section_sampler_id);
        }
    }

    void Project::set_show_sampler(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (this->state->debug.show_sampler == value) return;
        this->state->debug.show_sampler = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_sampler_points(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (this->state->debug.show_sampler_points == value) return;
        this->state->debug.show_sampler_points = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_sampler_rays(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (this->state->debug.show_sampler_rays == value) return;
        this->state->debug.show_sampler_rays = value;
        ++this->state->scene_revision;
    }

    void Project::set_sampler_point_radius(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"HyFluid sampler point radius must be finite and positive."};
        if (this->state->debug.sampler_point_radius == value) return;
        this->state->debug.sampler_point_radius = value;
        ++this->state->scene_revision;
    }

    void Project::set_sampler_ray_width(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"HyFluid project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"HyFluid sampler ray width must be finite and positive."};
        if (this->state->debug.sampler_ray_width == value) return;
        this->state->debug.sampler_ray_width = value;
        ++this->state->scene_revision;
    }
} // namespace hyfluid::project

extern "C" SPECTRA_SCENE_EXPORT auto spectra_scene_plugin_v16(void) -> decltype(hyfluid::plugin::export_plugin<hyfluid::project::Project>()) {
    return hyfluid::plugin::export_plugin<hyfluid::project::Project>();
}
