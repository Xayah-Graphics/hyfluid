module;
#include "hyfluid.train.h"

#include "hyfluid.train.config.h"

#include "json/json.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
module hyfluid.train;
import std;

namespace hyfluid::train {
    void HyFluid::initialize(const DatasetView& dataset) {
        try {
            if (!std::isfinite(dataset.scene_scale) || dataset.scene_scale <= 0.0f) throw std::runtime_error{"scene scale must be finite and positive."};
            if (!std::isfinite(dataset.near) || !std::isfinite(dataset.far) || dataset.near <= 0.0f || dataset.far <= dataset.near) throw std::runtime_error{"dynamic dataset near/far is invalid."};
            if (!std::isfinite(dataset.phi)) throw std::runtime_error{"dynamic dataset phi must be finite."};
            if (dataset.rotation_axis != 'Y' && dataset.rotation_axis != 'Z') throw std::runtime_error{"dynamic dataset rotation_axis must be Y or Z."};
            if (dataset.frame_sets.empty()) throw std::runtime_error{"dynamic dataset must contain at least one frame set."};
            if (dataset.videos.empty()) throw std::runtime_error{"dynamic dataset must contain video metadata."};
            if (dataset.sim_to_world.size() != 16uz) throw std::runtime_error{"sim_to_world must contain 16 values."};
            if (dataset.world_to_sim.size() != 16uz) throw std::runtime_error{"world_to_sim must contain 16 values."};
            if (dataset.voxel_scale.size() != 3uz) throw std::runtime_error{"voxel_scale must contain 3 values."};
            if (dataset.render_center.size() != 3uz) throw std::runtime_error{"render_center must contain 3 values."};
            for (std::size_t i = 0uz; i < 16uz; ++i) {
                if (!std::isfinite(dataset.sim_to_world[i])) throw std::runtime_error{"sim_to_world contains non-finite values."};
                if (!std::isfinite(dataset.world_to_sim[i])) throw std::runtime_error{"world_to_sim contains non-finite values."};
            }
            for (std::size_t i = 0uz; i < 3uz; ++i) {
                if (!std::isfinite(dataset.voxel_scale[i]) || dataset.voxel_scale[i] == 0.0f) throw std::runtime_error{"voxel_scale contains invalid values."};
                if (!std::isfinite(dataset.render_center[i])) throw std::runtime_error{"render_center contains non-finite values."};
            }
            constexpr std::array field_sim_extent = {
                config::scalar_real_active_sim_max[0u] - config::scalar_real_active_sim_min[0u],
                config::scalar_real_active_sim_max[1u] - config::scalar_real_active_sim_min[1u],
                config::scalar_real_active_sim_max[2u] - config::scalar_real_active_sim_min[2u],
            };
            for (const float extent : field_sim_extent)
                if (!std::isfinite(extent) || extent <= 0.0f) throw std::runtime_error{"ScalarReal active simulation box is invalid."};
            for (std::size_t row = 0uz; row < 3uz; ++row) {
                this->host.field_to_world_translation[row] = dataset.sim_to_world[row * 4uz + 3uz];
                for (std::size_t column = 0uz; column < 3uz; ++column) {
                    const float sim_coordinate = config::scalar_real_active_sim_min[column] * dataset.voxel_scale[column];
                    this->host.field_to_world_translation[row] += dataset.sim_to_world[row * 4uz + column] * sim_coordinate;
                    this->host.field_to_world_linear[row * 3uz + column] = dataset.sim_to_world[row * 4uz + column] * dataset.voxel_scale[column] * field_sim_extent[column];
                }
            }
            for (std::size_t column = 0uz; column < 3uz; ++column) {
                const float x      = this->host.field_to_world_linear[column];
                const float y      = this->host.field_to_world_linear[3uz + column];
                const float z      = this->host.field_to_world_linear[6uz + column];
                const float extent = std::sqrt(x * x + y * y + z * z);
                if (!std::isfinite(extent) || extent <= 0.0f) throw std::runtime_error{"Field domain metric extent is invalid."};
            }
            this->host.scene_scale   = dataset.scene_scale;
            this->host.near          = dataset.near;
            this->host.far           = dataset.far;
            this->host.phi           = dataset.phi;
            this->host.rotation_axis = dataset.rotation_axis;

            // ====================================================================================================
            // 1. UPLOAD FRAME SETS TO GPU
            // ====================================================================================================
            {
                this->host.frame_sets.reserve(dataset.frame_sets.size());
                this->device.frame_sets.reserve(dataset.frame_sets.size());
                for (const FrameSetView& frame_set : dataset.frame_sets) {
                    if (frame_set.name.empty()) throw std::runtime_error{"frame set name must not be empty."};
                    if (frame_set.view_count == 0u || frame_set.time_count == 0u) throw std::runtime_error{std::format("frame set '{}' must declare positive view_count and time_count.", frame_set.name)};
                    if (frame_set.frames.empty()) throw std::runtime_error{std::format("frame set '{}' contains no frames.", frame_set.name)};
                    if (frame_set.view_count > std::numeric_limits<std::uint32_t>::max() / frame_set.time_count) throw std::runtime_error{std::format("frame set '{}' view-time grid is too large.", frame_set.name)};
                    const std::uint32_t expected_frame_count = frame_set.view_count * frame_set.time_count;
                    if (frame_set.frames.size() != expected_frame_count) throw std::runtime_error{std::format("frame set '{}' contains {} frames; expected dense {}x{} grid.", frame_set.name, frame_set.frames.size(), frame_set.view_count, frame_set.time_count)};
                    for (const HostFrameSet& existing_frame_set : this->host.frame_sets)
                        if (existing_frame_set.name == frame_set.name) throw std::runtime_error{std::format("frame set '{}' was loaded more than once.", frame_set.name)};
                    if (this->host.frame_sets.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error{"dynamic dataset contains too many frame sets."};
                    if (this->host.frames.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error{"dynamic dataset contains too many frames."};
                    if (frame_set.frames.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) - this->host.frames.size()) throw std::runtime_error{"dynamic dataset contains too many frames."};

                    const auto host_frame_set_index = static_cast<std::uint32_t>(this->host.frame_sets.size());
                    const auto host_frame_offset    = static_cast<std::uint32_t>(this->host.frames.size());
                    const FrameView& first_frame    = frame_set.frames.front();
                    if (first_frame.width == 0u || first_frame.height == 0u) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid dimensions.", frame_set.name)};
                    if (first_frame.height > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(first_frame.width) / 4ull) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    const std::uint64_t frame_pixel_bytes = static_cast<std::uint64_t>(first_frame.width) * static_cast<std::uint64_t>(first_frame.height) * 4ull;
                    if (frame_pixel_bytes > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    if (frame_set.frames.size() > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(frame_pixel_bytes)) throw std::runtime_error{std::format("frame set '{}' pixel buffer is too large.", frame_set.name)};

                    std::vector<std::uint8_t> pixels;
                    std::vector<float> camera;
                    std::vector<float> intrinsics;
                    std::vector<float> times;
                    std::vector<std::uint32_t> view_indices;
                    std::vector<std::uint32_t> time_indices;
                    std::vector frame_indices(frame_set.frames.size(), std::numeric_limits<std::uint32_t>::max());
                    std::vector<std::array<float, 12u>> view_camera(frame_set.view_count);
                    std::vector<std::array<float, 4u>> view_intrinsics(frame_set.view_count);
                    std::vector<std::uint8_t> view_reference_seen(frame_set.view_count, 0u);
                    pixels.reserve(frame_set.frames.size() * static_cast<std::size_t>(frame_pixel_bytes));
                    camera.reserve(frame_set.frames.size() * 12uz);
                    intrinsics.reserve(frame_set.frames.size() * 4uz);
                    times.reserve(frame_set.frames.size());
                    view_indices.reserve(frame_set.frames.size());
                    time_indices.reserve(frame_set.frames.size());

                    std::uint32_t frame_index = 0u;
                    for (const FrameView& frame : frame_set.frames) {
                        if (frame.width != first_frame.width || frame.height != first_frame.height) throw std::runtime_error{std::format("frame set '{}' contains mixed image dimensions.", frame_set.name)};
                        if (frame.rgba.size() != static_cast<std::size_t>(frame_pixel_bytes)) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} RGBA bytes; expected {}.", frame_set.name, frame.rgba.size(), frame_pixel_bytes)};
                        if (frame.camera.size() != 12uz) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} camera values; expected 12.", frame_set.name, frame.camera.size())};
                        if (!std::isfinite(frame.focal_x) || !std::isfinite(frame.focal_y) || frame.focal_x <= 0.0f || frame.focal_y <= 0.0f) throw std::runtime_error{std::format("frame set '{}' contains invalid focal length.", frame_set.name)};
                        if (!std::isfinite(frame.principal_x) || !std::isfinite(frame.principal_y) || frame.principal_x < 0.0f || frame.principal_y < 0.0f || frame.principal_x >= static_cast<float>(frame.width) || frame.principal_y >= static_cast<float>(frame.height)) throw std::runtime_error{std::format("frame set '{}' contains invalid principal point.", frame_set.name)};
                        if (!std::isfinite(frame.time)) throw std::runtime_error{std::format("frame set '{}' contains non-finite time.", frame_set.name)};
                        if (frame.view_index >= frame_set.view_count) throw std::runtime_error{std::format("frame set '{}' contains view_index {} outside view_count {}.", frame_set.name, frame.view_index, frame_set.view_count)};
                        if (frame.time_index >= frame_set.time_count) throw std::runtime_error{std::format("frame set '{}' contains time_index {} outside time_count {}.", frame_set.name, frame.time_index, frame_set.time_count)};
                        for (const float camera_value : frame.camera)
                            if (!std::isfinite(camera_value)) throw std::runtime_error{std::format("frame set '{}' contains non-finite camera values.", frame_set.name)};
                        const std::array frame_intrinsics = {frame.focal_x, frame.focal_y, frame.principal_x, frame.principal_y};
                        if (view_reference_seen[frame.view_index] == 0u) {
                            for (std::size_t i = 0uz; i < 12uz; ++i) view_camera[frame.view_index][i] = frame.camera[i];
                            view_intrinsics[frame.view_index]     = frame_intrinsics;
                            view_reference_seen[frame.view_index] = 1u;
                        } else {
                            for (std::size_t i = 0uz; i < 12uz; ++i)
                                if (view_camera[frame.view_index][i] != frame.camera[i]) throw std::runtime_error{std::format("frame set '{}' contains changing camera values for view {}.", frame_set.name, frame.view_index)};
                            for (std::size_t i = 0uz; i < 4uz; ++i)
                                if (view_intrinsics[frame.view_index][i] != frame_intrinsics[i]) throw std::runtime_error{std::format("frame set '{}' contains changing intrinsics for view {}.", frame_set.name, frame.view_index)};
                        }

                        const std::uint32_t frame_grid_index = frame.view_index * frame_set.time_count + frame.time_index;
                        if (frame_indices[frame_grid_index] != std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains duplicate frame at view {} time {}.", frame_set.name, frame.view_index, frame.time_index)};
                        frame_indices[frame_grid_index] = frame_index;

                        const std::uint64_t pixel_offset = pixels.size();
                        pixels.append_range(frame.rgba);
                        camera.append_range(frame.camera);
                        intrinsics.push_back(frame.focal_x);
                        intrinsics.push_back(frame.focal_y);
                        intrinsics.push_back(frame.principal_x);
                        intrinsics.push_back(frame.principal_y);
                        times.push_back(frame.time);
                        view_indices.push_back(frame.view_index);
                        time_indices.push_back(frame.time_index);
                        this->host.frames.push_back(HostFrame{
                            .frame_set_index = host_frame_set_index,
                            .width           = frame.width,
                            .height          = frame.height,
                            .focal_x         = frame.focal_x,
                            .focal_y         = frame.focal_y,
                            .principal_x     = frame.principal_x,
                            .principal_y     = frame.principal_y,
                            .time            = frame.time,
                            .view_index      = frame.view_index,
                            .time_index      = frame.time_index,
                            .pixel_offset    = pixel_offset,
                        });
                        ++frame_index;
                    }
                    for (const std::uint32_t stored_frame_index : frame_indices)
                        if (stored_frame_index == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("frame set '{}' is missing at least one view-time frame.", frame_set.name)};

                    this->device.frame_sets.push_back({});
                    DeviceFrameSet& device_frame_set = this->device.frame_sets.back();
                    cuda::upload_dataset(pixels.data(), pixels.size(), camera.data(), camera.size(), intrinsics.data(), intrinsics.size(), times.data(), times.size(), view_indices.data(), view_indices.size(), time_indices.data(), time_indices.size(), frame_indices.data(), frame_indices.size(), device_frame_set.pixels, device_frame_set.camera, device_frame_set.intrinsics, device_frame_set.times, device_frame_set.view_indices, device_frame_set.time_indices, device_frame_set.frame_indices);

                    this->host.frame_sets.push_back(HostFrameSet{
                        .name              = std::string{frame_set.name},
                        .frame_offset      = host_frame_offset,
                        .frame_count       = expected_frame_count,
                        .view_count        = frame_set.view_count,
                        .time_count        = frame_set.time_count,
                        .width             = first_frame.width,
                        .height            = first_frame.height,
                        .pixel_bytes       = pixels.size(),
                        .camera_values     = camera.size(),
                        .intrinsics_values = intrinsics.size(),
                    });
                }
            }

            // ====================================================================================================
            // 2. VALIDATE VIDEO METADATA
            // ====================================================================================================
            {
                std::vector<std::vector<std::uint8_t>> video_seen_by_frame_set;
                video_seen_by_frame_set.reserve(this->host.frame_sets.size());
                std::vector video_count_by_frame_set(this->host.frame_sets.size(), 0u);
                for (const HostFrameSet& frame_set : this->host.frame_sets) video_seen_by_frame_set.emplace_back(frame_set.view_count, 0u);
                for (const VideoView& video : dataset.videos) {
                    if (video.frame_set.empty()) throw std::runtime_error{"video frame_set must not be empty."};
                    if (video.file_name.empty()) throw std::runtime_error{"video file_name must not be empty."};
                    std::uint32_t frame_set_index = std::numeric_limits<std::uint32_t>::max();
                    for (std::size_t i = 0uz; i < this->host.frame_sets.size(); ++i) {
                        if (this->host.frame_sets[i].name == video.frame_set) {
                            frame_set_index = static_cast<std::uint32_t>(i);
                            break;
                        }
                    }
                    if (frame_set_index == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("video '{}' references unloaded frame set '{}'.", video.file_name, video.frame_set)};
                    const HostFrameSet& host_frame_set = this->host.frame_sets[frame_set_index];
                    if (video.camera.size() != 12uz) throw std::runtime_error{std::format("video '{}' contains {} camera values; expected 12.", video.file_name, video.camera.size())};
                    if (video.width != host_frame_set.width || video.height != host_frame_set.height) throw std::runtime_error{std::format("video '{}' dimensions do not match frame set '{}'.", video.file_name, video.frame_set)};
                    if (video.frame_count != host_frame_set.time_count) throw std::runtime_error{std::format("video '{}' frame_count does not match frame set time_count.", video.file_name)};
                    if (video.frame_rate == 0u) throw std::runtime_error{std::format("video '{}' frame_rate must be positive.", video.file_name)};
                    if (video.view_index >= host_frame_set.view_count) throw std::runtime_error{std::format("video '{}' view_index is outside frame set view_count.", video.file_name)};
                    if (!std::isfinite(video.focal) || video.focal <= 0.0f) throw std::runtime_error{std::format("video '{}' focal must be finite and positive.", video.file_name)};
                    for (const float camera_value : video.camera)
                        if (!std::isfinite(camera_value)) throw std::runtime_error{std::format("video '{}' contains non-finite camera values.", video.file_name)};
                    if (video_seen_by_frame_set[frame_set_index][video.view_index] != 0u) throw std::runtime_error{std::format("frame set '{}' contains duplicate video view_index {}.", video.frame_set, video.view_index)};
                    video_seen_by_frame_set[frame_set_index][video.view_index] = 1u;
                    ++video_count_by_frame_set[frame_set_index];

                    HostVideo host_video{
                        .frame_set       = std::string{video.frame_set},
                        .file_name       = std::string{video.file_name},
                        .frame_set_index = frame_set_index,
                        .width           = video.width,
                        .height          = video.height,
                        .frame_count     = video.frame_count,
                        .frame_rate      = video.frame_rate,
                        .view_index      = video.view_index,
                        .focal           = video.focal,
                    };
                    for (std::size_t i = 0uz; i < 12uz; ++i) host_video.camera[i] = video.camera[i];
                    this->host.videos.push_back(std::move(host_video));
                }
                for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                    const HostFrameSet& frame_set = this->host.frame_sets[frame_set_index];
                    if (video_count_by_frame_set[frame_set_index] != frame_set.view_count) throw std::runtime_error{std::format("frame set '{}' must contain one video metadata entry per view.", frame_set.name)};
                }
            }

            std::uint64_t evaluation_pixel_capacity = 0ull;
            for (const HostFrameSet& frame_set : this->host.frame_sets) {
                if (frame_set.width % config::training_image_downsample != 0u || frame_set.height % config::training_image_downsample != 0u) throw std::runtime_error{std::format("frame set '{}' dimensions must be divisible by training image downsample.", frame_set.name)};
                const std::uint64_t render_width  = frame_set.width / config::training_image_downsample;
                const std::uint64_t render_height = frame_set.height / config::training_image_downsample;
                evaluation_pixel_capacity         = std::max(evaluation_pixel_capacity, render_width * render_height);
            }
            if (evaluation_pixel_capacity == 0ull || evaluation_pixel_capacity > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"evaluation render pixel capacity is invalid."};
            this->host.evaluation_pixel_capacity = static_cast<std::uint32_t>(evaluation_pixel_capacity);

            cuda::upload_field_domain(this->host.field_to_world_linear.data(), this->device.field_to_world_linear);
            cuda::allocate_sampler_buffers(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.occupancy_grid_occupied_count);
            cuda::allocate_training_loss_buffers(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.network_output_gradients);
            cuda::allocate_network_buffers(this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.network_input_gradients, this->device.network_hidden_gradients, this->device.cublaslt_handle, this->device.cublaslt_workspace);
            cuda::allocate_trainable_parameter_buffers(this->device.params_full_precision, this->device.params, this->device.param_gradients);
            cuda::initialize_trainable_parameters(this->device.params_full_precision, this->device.params);
            cuda::allocate_optimizer_buffers(this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            cuda::allocate_evaluation_buffers(this->host.evaluation_pixel_capacity, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.evaluation_pixels);
            cuda::set_occupancy_grid_full(this->device.occupancy, this->device.occupancy_grid_occupied_count);
            this->host.current_step                            = 0u;
            this->host.rays_per_batch                          = config::initial_rays_per_batch;
            this->host.inference_sample_count                  = config::network_batch_size;
            this->host.measured_sample_count_before_compaction = 0u;
            this->host.measured_sample_count                   = 0u;
            this->host.occupancy_grid_occupied_cells           = config::nerf_grid_cells;
        } catch (...) {
            cuda::destroy_network_handle(this->device.cublaslt_handle);
            cuda::free_device_buffers(this->device.field_to_world_linear, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.occupancy_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.network_output_gradients, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.network_input_gradients, this->device.network_hidden_gradients, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.evaluation_pixels);
            for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.times, frame_set.view_indices, frame_set.time_indices, frame_set.frame_indices);
            throw;
        }
    }

    HyFluid::~HyFluid() noexcept {
        cuda::destroy_network_handle(this->device.cublaslt_handle);
        cuda::free_device_buffers(this->device.field_to_world_linear, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.occupancy_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.network_output_gradients, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.network_input_gradients, this->device.network_hidden_gradients, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.evaluation_pixels);
        for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.times, frame_set.view_indices, frame_set.time_indices, frame_set.frame_indices);
    }

    std::expected<OptimizationStats, std::string> HyFluid::optimize(const OptimizationRequest& request) {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"optimization frame set must not be empty."};
            if (request.iterations < 1) throw std::runtime_error{"optimization iterations must be positive."};
            const HostFrameSet* host_frame_set     = nullptr;
            const DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                if (this->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set   = std::addressof(this->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("optimization frame set '{}' is not loaded.", request.frame_set)};

            const auto optimize_start            = std::chrono::steady_clock::now();
            std::uint32_t ray_count              = 0u;
            std::uint32_t sample_count           = 0u;
            std::uint32_t compacted_sample_count = 0u;
            std::uint32_t loss_value_count       = this->host.rays_per_batch;
            for (std::int32_t i = 0; i < request.iterations; ++i) {
                loss_value_count = this->host.rays_per_batch;
                cuda::sample_training_batch(device_frame_set->camera, device_frame_set->intrinsics, device_frame_set->frame_indices, this->device.field_to_world_linear, host_frame_set->view_count, host_frame_set->time_count, host_frame_set->width, host_frame_set->height, this->host.current_step, this->host.rays_per_batch, this->host.inference_sample_count, this->device.occupancy, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter);
                cuda::evaluate_network(this->host.inference_sample_count, this->device.sample_coords, this->device.params, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::compute_training_loss_and_compact_samples(this->host.rays_per_batch, this->host.current_step, this->device.ray_counter, device_frame_set->pixels, device_frame_set->frame_indices, host_frame_set->view_count, host_frame_set->time_count, host_frame_set->width, host_frame_set->height, this->device.network_output, this->device.compacted_sample_counter, this->device.ray_indices, this->device.numsteps, this->device.sample_coords, this->device.compacted_sample_coords, this->device.network_output_gradients, this->device.param_gradients, this->device.params, this->device.loss_values);
                cuda::pad_compacted_training_batch(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.network_output_gradients);
                cuda::forward_network(this->device.compacted_sample_coords, this->device.params, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::backward_network(this->device.compacted_sample_coords, this->device.params, this->device.param_gradients, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.network_output_gradients, this->device.network_input_gradients, this->device.network_hidden_gradients, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::step_optimizer(this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                cuda::read_counter(this->device.ray_counter, ray_count);
                cuda::read_counter(this->device.sample_counter, sample_count);
                if (sample_count > config::max_samples) sample_count = config::max_samples;
                cuda::read_counter(this->device.compacted_sample_counter, compacted_sample_count);
                if (compacted_sample_count > config::network_batch_size) compacted_sample_count = config::network_batch_size;
                if (ray_count == 0u || sample_count == 0u || compacted_sample_count == 0u) throw std::runtime_error{"HyFluid density training produced an empty batch."};
                this->host.measured_sample_count_before_compaction = sample_count;
                this->host.measured_sample_count                   = compacted_sample_count;
                this->host.inference_sample_count                  = ((std::min(sample_count, config::max_samples) + config::network_batch_granularity - 1u) / config::network_batch_granularity) * config::network_batch_granularity;
                if (this->host.inference_sample_count == 0u) this->host.inference_sample_count = config::network_batch_granularity;
                if (this->host.inference_sample_count > config::network_batch_size) this->host.inference_sample_count = config::network_batch_size;
                ++this->host.current_step;
            }
            float loss_sum = 0.0f;
            cuda::read_loss_sum(this->device.loss_values, loss_value_count, loss_sum);
            cuda::read_counter(this->device.occupancy_grid_occupied_count, this->host.occupancy_grid_occupied_cells);
            const float psnr = loss_sum > 0.0f ? -10.0f * std::log10(loss_sum) : std::numeric_limits<float>::infinity();
            return OptimizationStats{
                .step                           = this->host.current_step,
                .rays_per_batch                 = this->host.rays_per_batch,
                .ray_count                      = ray_count,
                .sample_count_before_compaction = sample_count,
                .sample_count                   = compacted_sample_count,
                .occupancy_grid_occupied_cells  = this->host.occupancy_grid_occupied_cells,
                .loss                           = loss_sum,
                .psnr                           = psnr,
                .elapsed_ms                     = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - optimize_start).count(),
                .sample_efficiency_ratio        = sample_count == 0u ? 0.0f : static_cast<float>(compacted_sample_count) / static_cast<float>(sample_count),
                .occupancy_grid_ratio           = static_cast<float>(this->host.occupancy_grid_occupied_cells) / static_cast<float>(config::nerf_grid_cells),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<EvaluationStats, std::string> HyFluid::evaluate(const EvaluationRequest& request) const {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"evaluation frame set must not be empty."};
            if (request.output_dir.empty()) throw std::runtime_error{"evaluation output directory must not be empty."};
            if (std::filesystem::exists(request.output_dir) && !std::filesystem::is_directory(request.output_dir)) throw std::runtime_error{std::format("evaluation output path '{}' is not a directory.", request.output_dir.string())};
            std::filesystem::create_directories(request.output_dir);
            if (!std::filesystem::is_directory(request.output_dir)) throw std::runtime_error{std::format("failed to create evaluation output directory '{}'.", request.output_dir.string())};

            const HostFrameSet* host_frame_set     = nullptr;
            const DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                if (this->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set   = std::addressof(this->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("evaluation frame set '{}' is not loaded.", request.frame_set)};
            if (host_frame_set->view_count == 0u || host_frame_set->time_count == 0u || host_frame_set->frame_count != host_frame_set->view_count * host_frame_set->time_count) throw std::runtime_error{std::format("evaluation frame set '{}' is not a dense view-time grid.", host_frame_set->name)};
            if (host_frame_set->width % config::training_image_downsample != 0u || host_frame_set->height % config::training_image_downsample != 0u) throw std::runtime_error{std::format("evaluation frame set '{}' dimensions are not divisible by training image downsample.", host_frame_set->name)};

            const std::uint32_t render_width  = host_frame_set->width / config::training_image_downsample;
            const std::uint32_t render_height = host_frame_set->height / config::training_image_downsample;
            if (render_width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) || render_height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) throw std::runtime_error{"evaluation render dimensions exceed PNG writer limits."};
            const std::uint64_t render_pixels = static_cast<std::uint64_t>(render_width) * render_height;
            if (render_pixels == 0ull || render_pixels > this->host.evaluation_pixel_capacity) throw std::runtime_error{"evaluation render image exceeds allocated capacity."};
            const std::uint64_t image_count = static_cast<std::uint64_t>(host_frame_set->view_count) * host_frame_set->time_count;
            if (image_count > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"evaluation image count is too large."};

            const auto evaluation_start = std::chrono::steady_clock::now();
            std::vector<std::uint8_t> evaluation_pixels(static_cast<std::size_t>(render_pixels) * 3uz);
            double loss_sum                    = 0.0;
            std::uint32_t rendered_image_count = 0u;
            for (std::uint32_t view_index = 0u; view_index < host_frame_set->view_count; ++view_index) {
                const std::filesystem::path view_output_dir = request.output_dir / std::format("view_{:04}", view_index);
                std::filesystem::create_directories(view_output_dir);
                if (!std::filesystem::is_directory(view_output_dir)) throw std::runtime_error{std::format("failed to create evaluation view output directory '{}'.", view_output_dir.string())};
                for (std::uint32_t time_index = 0u; time_index < host_frame_set->time_count; ++time_index) {
                    double image_loss_sum = 0.0;
                    cuda::run_evaluation_image(device_frame_set->pixels, device_frame_set->camera, device_frame_set->intrinsics, device_frame_set->frame_indices, this->device.field_to_world_linear, host_frame_set->view_count, host_frame_set->time_count, host_frame_set->width, host_frame_set->height, view_index, time_index, this->host.evaluation_pixel_capacity, this->device.occupancy, this->device.params, this->device.sample_coords, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.cublaslt_handle, this->device.cublaslt_workspace, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.evaluation_pixels, evaluation_pixels.data(), image_loss_sum);
                    loss_sum += image_loss_sum;
                    const std::filesystem::path output_path = view_output_dir / std::format("rgb_{:04}.png", time_index);
                    const std::string output_path_text      = output_path.string();
                    if (stbi_write_png(output_path_text.c_str(), static_cast<int>(render_width), static_cast<int>(render_height), 3, evaluation_pixels.data(), static_cast<int>(render_width * 3u)) == 0) throw std::runtime_error{std::format("failed to write evaluation image '{}'.", output_path_text)};
                    ++rendered_image_count;
                }
            }

            const std::uint64_t pixel_count = render_pixels * rendered_image_count;
            if (pixel_count == 0ull) throw std::runtime_error{"evaluation produced no pixels."};
            const double mse = loss_sum / (static_cast<double>(pixel_count) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"evaluation produced non-finite MSE."};
            const float psnr = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity();
            EvaluationStats stats{
                .frame_set            = std::string{request.frame_set},
                .step                 = this->host.current_step,
                .render_width         = render_width,
                .render_height        = render_height,
                .image_count          = static_cast<std::uint32_t>(image_count),
                .rendered_image_count = rendered_image_count,
                .pixel_count          = pixel_count,
                .mse                  = static_cast<float>(mse),
                .psnr                 = psnr,
                .elapsed_ms           = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - evaluation_start).count(),
                .output_dir           = request.output_dir,
            };

            const std::filesystem::path metrics_path = request.output_dir / "metrics.json";
            std::ofstream metrics{metrics_path, std::ios::binary | std::ios::trunc};
            if (!metrics) throw std::runtime_error{std::format("failed to open evaluation metrics '{}'.", metrics_path.string())};
            metrics << "{\n";
            metrics << "  \"frame_set\": " << std::quoted(stats.frame_set) << ",\n";
            metrics << "  \"step\": " << stats.step << ",\n";
            metrics << "  \"render_width\": " << stats.render_width << ",\n";
            metrics << "  \"render_height\": " << stats.render_height << ",\n";
            metrics << "  \"view_count\": " << host_frame_set->view_count << ",\n";
            metrics << "  \"time_count\": " << host_frame_set->time_count << ",\n";
            metrics << "  \"image_count\": " << stats.image_count << ",\n";
            metrics << "  \"rendered_image_count\": " << stats.rendered_image_count << ",\n";
            metrics << "  \"pixel_count\": " << stats.pixel_count << ",\n";
            metrics << "  \"mse\": " << std::setprecision(9) << stats.mse << ",\n";
            metrics << "  \"psnr\": " << std::setprecision(9) << stats.psnr << ",\n";
            metrics << "  \"elapsed_ms\": " << std::setprecision(9) << stats.elapsed_ms << ",\n";
            metrics << "  \"output_dir\": " << std::quoted(stats.output_dir.string()) << "\n";
            metrics << "}\n";
            return stats;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluid::export_weights(const std::filesystem::path& path) const {
        try {
            static_assert(std::endian::native == std::endian::little);
            if (path.empty()) throw std::runtime_error{"weights export path must not be empty."};
            if (!path.parent_path().empty() && !std::filesystem::is_directory(path.parent_path())) throw std::runtime_error{std::format("weights export parent directory '{}' does not exist.", path.parent_path().string())};
            if (this->device.params_full_precision == nullptr) throw std::runtime_error{"trainable parameters are not initialized."};

            struct SafetensorsTensor final {
                std::string_view name;
                std::uint32_t param_offset;
                std::uint64_t rows;
                std::uint64_t cols;
            };

            constexpr std::array tensors = std::to_array<SafetensorsTensor>({
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = config::network_parameter_layout.mlp_input_weight_offset, .rows = config::mlp_width, .cols = config::mlp_input_width},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = config::network_parameter_layout.mlp_output_weight_offset, .rows = config::network_output_width, .cols = config::mlp_width},
                SafetensorsTensor{.name = "density_global_rgb.param", .param_offset = config::network_parameter_layout.global_rgb_offset, .rows = 1u, .cols = 1u},
                SafetensorsTensor{.name = "hash4.params", .param_offset = config::network_parameter_layout.hash4_param_offset, .rows = config::network_parameter_layout.hash4_param_count / config::hash4_features_per_level, .cols = config::hash4_features_per_level},
            });

            std::vector<float> host_params(config::network_parameter_layout.total_param_count);
            cuda::download_trainable_parameters(this->device.params_full_precision, host_params.data());

            std::string hash4_resolutions_text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count; ++i) {
                if (!hash4_resolutions_text.empty()) hash4_resolutions_text += ",";
                hash4_resolutions_text += std::format("{}", config::hash4_resolutions[i]);
            }

            std::string hash4_offsets_text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count + 1u; ++i) {
                if (!hash4_offsets_text.empty()) hash4_offsets_text += ",";
                hash4_offsets_text += std::format("{}", config::network_parameter_layout.hash4_offsets[i]);
            }

            nlohmann::json metadata              = nlohmann::json::object();
            metadata["format"]                   = "hyfluid.density.weights.v1";
            metadata["stage"]                    = "density";
            metadata["architecture_fingerprint"] = std::format("hash4:l{}:f{}:max{}:res{}:offsets{}|mlp:w{}:in{}:out{}:global{}:total{}|domain:min{:.9g},{:.9g},{:.9g}:max{:.9g},{:.9g},{:.9g}", config::hash4_level_count, config::hash4_features_per_level, config::hash4_max_entries, hash4_resolutions_text, hash4_offsets_text, config::mlp_width, config::mlp_input_width, config::network_output_width, config::network_parameter_layout.global_rgb_offset, config::network_parameter_layout.total_param_count, config::scalar_real_active_sim_min[0u], config::scalar_real_active_sim_min[1u], config::scalar_real_active_sim_min[2u], config::scalar_real_active_sim_max[0u], config::scalar_real_active_sim_max[1u], config::scalar_real_active_sim_max[2u]);
            metadata["hash4_level_count"]        = std::format("{}", config::hash4_level_count);
            metadata["hash4_features_per_level"] = std::format("{}", config::hash4_features_per_level);
            metadata["hash4_max_entries"]        = std::format("{}", config::hash4_max_entries);
            metadata["hash4_resolutions"]        = hash4_resolutions_text;
            metadata["hash4_offsets"]            = hash4_offsets_text;
            metadata["mlp_width"]                = std::format("{}", config::mlp_width);
            metadata["mlp_input_width"]          = std::format("{}", config::mlp_input_width);
            metadata["network_output_width"]     = std::format("{}", config::network_output_width);
            metadata["global_rgb_offset"]        = std::format("{}", config::network_parameter_layout.global_rgb_offset);
            metadata["total_param_count"]        = std::format("{}", config::network_parameter_layout.total_param_count);
            metadata["sample_coord_floats"]      = std::format("{}", config::sample_coord_floats);
            metadata["active_sim_min"]           = std::format("{:.9g},{:.9g},{:.9g}", config::scalar_real_active_sim_min[0u], config::scalar_real_active_sim_min[1u], config::scalar_real_active_sim_min[2u]);
            metadata["active_sim_max"]           = std::format("{:.9g},{:.9g},{:.9g}", config::scalar_real_active_sim_max[0u], config::scalar_real_active_sim_max[1u], config::scalar_real_active_sim_max[2u]);

            nlohmann::json header  = nlohmann::json::object();
            header["__metadata__"] = metadata;

            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::uint64_t byte_count   = tensor.rows * tensor.cols * sizeof(float);
                header[std::string{tensor.name}] = nlohmann::json{
                    {"dtype", "F32"},
                    {"shape", nlohmann::json::array({tensor.rows, tensor.cols})},
                    {"data_offsets", nlohmann::json::array({data_offset, data_offset + byte_count})},
                };
                data_offset += byte_count;
            }

            const std::string header_text   = header.dump();
            const std::uint64_t header_size = header_text.size();
            std::ofstream output{path, std::ios::binary | std::ios::trunc};
            if (!output) throw std::runtime_error{std::format("failed to open weights export path '{}'.", path.string())};

            output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
            output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            for (const SafetensorsTensor& tensor : tensors) output.write(reinterpret_cast<const char*>(host_params.data() + tensor.param_offset), static_cast<std::streamsize>(tensor.rows * tensor.cols * sizeof(float)));
            if (!output) throw std::runtime_error{std::format("failed to write weights file '{}'.", path.string())};

            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluid::load_weights(const std::filesystem::path& path) const {
        try {
            static_assert(std::endian::native == std::endian::little);
            if (this->host.current_step != 0u) throw std::runtime_error{"weights can only be loaded before training starts."};
            if (path.empty()) throw std::runtime_error{"weights load path must not be empty."};
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error{std::format("weights file '{}' does not exist.", path.string())};
            if (this->device.params_full_precision == nullptr) throw std::runtime_error{"trainable parameters are not initialized."};

            struct SafetensorsTensor final {
                std::string_view name;
                std::uint32_t param_offset;
                std::uint64_t rows;
                std::uint64_t cols;
            };

            constexpr std::array tensors = std::to_array<SafetensorsTensor>({
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = config::network_parameter_layout.mlp_input_weight_offset, .rows = config::mlp_width, .cols = config::mlp_input_width},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = config::network_parameter_layout.mlp_output_weight_offset, .rows = config::network_output_width, .cols = config::mlp_width},
                SafetensorsTensor{.name = "density_global_rgb.param", .param_offset = config::network_parameter_layout.global_rgb_offset, .rows = 1u, .cols = 1u},
                SafetensorsTensor{.name = "hash4.params", .param_offset = config::network_parameter_layout.hash4_param_offset, .rows = config::network_parameter_layout.hash4_param_count / config::hash4_features_per_level, .cols = config::hash4_features_per_level},
            });

            std::string hash4_resolutions_text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count; ++i) {
                if (!hash4_resolutions_text.empty()) hash4_resolutions_text += ",";
                hash4_resolutions_text += std::format("{}", config::hash4_resolutions[i]);
            }

            std::string hash4_offsets_text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count + 1u; ++i) {
                if (!hash4_offsets_text.empty()) hash4_offsets_text += ",";
                hash4_offsets_text += std::format("{}", config::network_parameter_layout.hash4_offsets[i]);
            }

            nlohmann::json expected_metadata              = nlohmann::json::object();
            expected_metadata["format"]                   = "hyfluid.density.weights.v1";
            expected_metadata["stage"]                    = "density";
            expected_metadata["architecture_fingerprint"] = std::format("hash4:l{}:f{}:max{}:res{}:offsets{}|mlp:w{}:in{}:out{}:global{}:total{}|domain:min{:.9g},{:.9g},{:.9g}:max{:.9g},{:.9g},{:.9g}", config::hash4_level_count, config::hash4_features_per_level, config::hash4_max_entries, hash4_resolutions_text, hash4_offsets_text, config::mlp_width, config::mlp_input_width, config::network_output_width, config::network_parameter_layout.global_rgb_offset, config::network_parameter_layout.total_param_count, config::scalar_real_active_sim_min[0u], config::scalar_real_active_sim_min[1u], config::scalar_real_active_sim_min[2u], config::scalar_real_active_sim_max[0u], config::scalar_real_active_sim_max[1u], config::scalar_real_active_sim_max[2u]);
            expected_metadata["hash4_level_count"]        = std::format("{}", config::hash4_level_count);
            expected_metadata["hash4_features_per_level"] = std::format("{}", config::hash4_features_per_level);
            expected_metadata["hash4_max_entries"]        = std::format("{}", config::hash4_max_entries);
            expected_metadata["hash4_resolutions"]        = hash4_resolutions_text;
            expected_metadata["hash4_offsets"]            = hash4_offsets_text;
            expected_metadata["mlp_width"]                = std::format("{}", config::mlp_width);
            expected_metadata["mlp_input_width"]          = std::format("{}", config::mlp_input_width);
            expected_metadata["network_output_width"]     = std::format("{}", config::network_output_width);
            expected_metadata["global_rgb_offset"]        = std::format("{}", config::network_parameter_layout.global_rgb_offset);
            expected_metadata["total_param_count"]        = std::format("{}", config::network_parameter_layout.total_param_count);
            expected_metadata["sample_coord_floats"]      = std::format("{}", config::sample_coord_floats);
            expected_metadata["active_sim_min"]           = std::format("{:.9g},{:.9g},{:.9g}", config::scalar_real_active_sim_min[0u], config::scalar_real_active_sim_min[1u], config::scalar_real_active_sim_min[2u]);
            expected_metadata["active_sim_max"]           = std::format("{:.9g},{:.9g},{:.9g}", config::scalar_real_active_sim_max[0u], config::scalar_real_active_sim_max[1u], config::scalar_real_active_sim_max[2u]);

            const std::uintmax_t file_size = std::filesystem::file_size(path);
            if (file_size < sizeof(std::uint64_t)) throw std::runtime_error{"weights file is too small for a safetensors header."};

            std::ifstream input{path, std::ios::binary};
            if (!input) throw std::runtime_error{std::format("failed to open weights file '{}'.", path.string())};

            std::uint64_t header_size = 0u;
            input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
            if (!input) throw std::runtime_error{"failed to read safetensors header length."};
            if (header_size == 0u || header_size > 100ull * 1024ull * 1024ull) throw std::runtime_error{"invalid safetensors header length."};
            if (sizeof(std::uint64_t) + header_size > file_size) throw std::runtime_error{"safetensors header length exceeds file size."};

            std::string header_text(header_size, '\0');
            input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            if (!input) throw std::runtime_error{"failed to read safetensors header."};
            if (header_text.empty() || header_text.front() != '{') throw std::runtime_error{"safetensors header must begin with '{'."};

            const nlohmann::json header = nlohmann::json::parse(header_text);
            if (!header.is_object()) throw std::runtime_error{"safetensors header must be a JSON object."};
            if (header.size() != tensors.size() + 1uz) throw std::runtime_error{"safetensors header contains unexpected tensors."};
            if (!header.contains("__metadata__") || !header.at("__metadata__").is_object()) throw std::runtime_error{"safetensors metadata is missing."};
            if (header.at("__metadata__") != expected_metadata) throw std::runtime_error{"safetensors metadata does not match the current HyFluid density configuration."};

            std::uint64_t expected_data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::string tensor_name{tensor.name};
                if (!header.contains(tensor_name)) throw std::runtime_error{std::format("safetensors tensor '{}' is missing.", tensor_name)};
                const nlohmann::json& tensor_header = header.at(tensor_name);
                if (!tensor_header.is_object() || tensor_header.size() != 3uz) throw std::runtime_error{std::format("safetensors tensor '{}' has an invalid header.", tensor_name)};
                if (!tensor_header.contains("dtype") || !tensor_header.at("dtype").is_string() || tensor_header.at("dtype").get<std::string>() != "F32") throw std::runtime_error{std::format("safetensors tensor '{}' must use dtype F32.", tensor_name)};
                if (!tensor_header.contains("shape") || !tensor_header.at("shape").is_array() || tensor_header.at("shape").size() != 2uz) throw std::runtime_error{std::format("safetensors tensor '{}' has an invalid shape.", tensor_name)};
                if ((!tensor_header.at("shape").at(0uz).is_number_integer() && !tensor_header.at("shape").at(0uz).is_number_unsigned()) || (!tensor_header.at("shape").at(1uz).is_number_integer() && !tensor_header.at("shape").at(1uz).is_number_unsigned())) throw std::runtime_error{std::format("safetensors tensor '{}' shape must contain integer dimensions.", tensor_name)};
                const std::int64_t actual_rows = tensor_header.at("shape").at(0uz).get<std::int64_t>();
                const std::int64_t actual_cols = tensor_header.at("shape").at(1uz).get<std::int64_t>();
                if (actual_rows < 0 || actual_cols < 0 || static_cast<std::uint64_t>(actual_rows) != tensor.rows || static_cast<std::uint64_t>(actual_cols) != tensor.cols) throw std::runtime_error{std::format("safetensors tensor '{}' shape mismatch.", tensor_name)};
                if (!tensor_header.contains("data_offsets") || !tensor_header.at("data_offsets").is_array() || tensor_header.at("data_offsets").size() != 2uz) throw std::runtime_error{std::format("safetensors tensor '{}' has invalid data_offsets.", tensor_name)};
                if ((!tensor_header.at("data_offsets").at(0uz).is_number_integer() && !tensor_header.at("data_offsets").at(0uz).is_number_unsigned()) || (!tensor_header.at("data_offsets").at(1uz).is_number_integer() && !tensor_header.at("data_offsets").at(1uz).is_number_unsigned())) throw std::runtime_error{std::format("safetensors tensor '{}' offsets must be integers.", tensor_name)};
                const std::int64_t actual_begin_signed = tensor_header.at("data_offsets").at(0uz).get<std::int64_t>();
                const std::int64_t actual_end_signed   = tensor_header.at("data_offsets").at(1uz).get<std::int64_t>();
                if (actual_begin_signed < 0 || actual_end_signed < 0) throw std::runtime_error{std::format("safetensors tensor '{}' offsets must be non-negative.", tensor_name)};
                const auto actual_begin        = static_cast<std::uint64_t>(actual_begin_signed);
                const auto actual_end          = static_cast<std::uint64_t>(actual_end_signed);
                const std::uint64_t byte_count = tensor.rows * tensor.cols * sizeof(float);
                if (actual_begin != expected_data_offset || actual_end != actual_begin + byte_count) throw std::runtime_error{std::format("safetensors tensor '{}' data_offsets mismatch.", tensor_name)};
                expected_data_offset += byte_count;
            }

            const std::uint64_t file_data_size = file_size - sizeof(std::uint64_t) - header_size;
            if (expected_data_offset != file_data_size) throw std::runtime_error{"safetensors data buffer size does not match tensor offsets."};

            std::vector<char> data(file_data_size);
            if (!data.empty()) input.read(data.data(), static_cast<std::streamsize>(data.size()));
            if (!input) throw std::runtime_error{"failed to read safetensors tensor data."};

            std::vector host_params(config::network_parameter_layout.total_param_count, 0.0f);
            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::uint64_t byte_count = tensor.rows * tensor.cols * sizeof(float);
                std::memcpy(host_params.data() + tensor.param_offset, data.data() + data_offset, byte_count);
                data_offset += byte_count;
            }

            for (const float value : host_params)
                if (!std::isfinite(value)) throw std::runtime_error{"weights file contains non-finite values."};

            cuda::upload_trainable_parameters(host_params.data(), this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::train
