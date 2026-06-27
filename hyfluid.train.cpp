module;
#include "hyfluid.train.h"
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
                this->host.sim_to_world[i] = dataset.sim_to_world[i];
                this->host.world_to_sim[i] = dataset.world_to_sim[i];
            }
            for (std::size_t i = 0uz; i < 3uz; ++i) {
                if (!std::isfinite(dataset.voxel_scale[i]) || dataset.voxel_scale[i] == 0.0f) throw std::runtime_error{"voxel_scale contains invalid values."};
                if (!std::isfinite(dataset.render_center[i])) throw std::runtime_error{"render_center contains non-finite values."};
                this->host.voxel_scale[i] = dataset.voxel_scale[i];
                this->host.render_center[i] = dataset.render_center[i];
            }
            this->host.scene_scale = dataset.scene_scale;
            this->host.near = dataset.near;
            this->host.far = dataset.far;
            this->host.phi = dataset.phi;
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

                    const std::uint32_t host_frame_set_index = static_cast<std::uint32_t>(this->host.frame_sets.size());
                    const std::uint32_t host_frame_offset = static_cast<std::uint32_t>(this->host.frames.size());
                    const FrameView& first_frame = frame_set.frames.front();
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
                    std::vector<std::uint32_t> frame_indices(frame_set.frames.size(), std::numeric_limits<std::uint32_t>::max());
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
                            .width = frame.width,
                            .height = frame.height,
                            .focal_x = frame.focal_x,
                            .focal_y = frame.focal_y,
                            .principal_x = frame.principal_x,
                            .principal_y = frame.principal_y,
                            .time = frame.time,
                            .view_index = frame.view_index,
                            .time_index = frame.time_index,
                            .pixel_offset = pixel_offset,
                        });
                        ++frame_index;
                    }
                    for (const std::uint32_t stored_frame_index : frame_indices)
                        if (stored_frame_index == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("frame set '{}' is missing at least one view-time frame.", frame_set.name)};

                    this->device.frame_sets.push_back({});
                    DeviceFrameSet& device_frame_set = this->device.frame_sets.back();
                    cuda::upload_dataset(pixels.data(), pixels.size(), camera.data(), camera.size(), intrinsics.data(), intrinsics.size(), times.data(), times.size(), view_indices.data(), view_indices.size(), time_indices.data(), time_indices.size(), frame_indices.data(), frame_indices.size(), device_frame_set.pixels, device_frame_set.camera, device_frame_set.intrinsics, device_frame_set.times, device_frame_set.view_indices, device_frame_set.time_indices, device_frame_set.frame_indices);

                    this->host.frame_sets.push_back(HostFrameSet{
                        .name = std::string{frame_set.name},
                        .frame_offset = host_frame_offset,
                        .frame_count = expected_frame_count,
                        .view_count = frame_set.view_count,
                        .time_count = frame_set.time_count,
                        .width = first_frame.width,
                        .height = first_frame.height,
                        .pixel_bytes = pixels.size(),
                        .camera_values = camera.size(),
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
                std::vector<std::uint32_t> video_count_by_frame_set(this->host.frame_sets.size(), 0u);
                for (const HostFrameSet& frame_set : this->host.frame_sets) video_seen_by_frame_set.push_back(std::vector<std::uint8_t>(frame_set.view_count, 0u));
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
                        .frame_set = std::string{video.frame_set},
                        .file_name = std::string{video.file_name},
                        .frame_set_index = frame_set_index,
                        .width = video.width,
                        .height = video.height,
                        .frame_count = video.frame_count,
                        .frame_rate = video.frame_rate,
                        .view_index = video.view_index,
                        .focal = video.focal,
                    };
                    for (std::size_t i = 0uz; i < 12uz; ++i) host_video.camera[i] = video.camera[i];
                    this->host.videos.push_back(std::move(host_video));
                }
                for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                    const HostFrameSet& frame_set = this->host.frame_sets[frame_set_index];
                    if (video_count_by_frame_set[frame_set_index] != frame_set.view_count) throw std::runtime_error{std::format("frame set '{}' must contain one video metadata entry per view.", frame_set.name)};
                }
            }

            this->host.current_step = 0u;
        } catch (...) {
            for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.times, frame_set.view_indices, frame_set.time_indices, frame_set.frame_indices);
            throw;
        }
    }

    HyFluid::~HyFluid() noexcept {
        for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.times, frame_set.view_indices, frame_set.time_indices, frame_set.frame_indices);
    }

    std::expected<OptimizationStats, std::string> HyFluid::optimize(const OptimizationRequest request) {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"optimization frame set must not be empty."};
            if (request.iterations < 1) throw std::runtime_error{"optimization iterations must be positive."};
            throw std::runtime_error{"HyFluid optimization is not implemented yet."};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<EvaluationStats, std::string> HyFluid::evaluate(const EvaluationRequest request) const {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"evaluation frame set must not be empty."};
            throw std::runtime_error{"HyFluid evaluation is not implemented yet."};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluid::export_weights(const std::filesystem::path& path) const {
        try {
            if (path.empty()) throw std::runtime_error{"weights export path must not be empty."};
            throw std::runtime_error{"HyFluid weights export is not implemented yet."};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluid::load_weights(const std::filesystem::path& path) {
        try {
            if (path.empty()) throw std::runtime_error{"weights load path must not be empty."};
            throw std::runtime_error{"HyFluid weights load is not implemented yet."};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::train
