module;
#include "hyfluid.train.h"
module hyfluid.train;
import std;

namespace hyfluid::train {
    void HyFluid::initialize(const std::span<const FrameSetView> frame_sets, const float scene_scale) {
        try {
            if (!std::isfinite(scene_scale) || scene_scale <= 0.0f) throw std::runtime_error{"scene scale must be finite and positive."};
            if (frame_sets.empty()) throw std::runtime_error{"dataset must contain at least one loaded frame set."};
            this->host.scene_scale = scene_scale;
            this->host.current_step = 0u;
            this->host.frame_sets.reserve(frame_sets.size());
            this->device.frame_sets.reserve(frame_sets.size());

            for (const FrameSetView& frame_set : frame_sets) {
                if (frame_set.name.empty()) throw std::runtime_error{"frame set name must not be empty."};
                if (frame_set.frames.empty()) throw std::runtime_error{std::format("frame set '{}' contains no frames.", frame_set.name)};
                for (const HostFrameSet& existing_frame_set : this->host.frame_sets)
                    if (existing_frame_set.name == frame_set.name) throw std::runtime_error{std::format("frame set '{}' was loaded more than once.", frame_set.name)};

                const FrameView& first_frame = frame_set.frames.front();
                if (first_frame.width == 0u || first_frame.height == 0u) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid dimensions.", frame_set.name)};
                if (!std::isfinite(first_frame.focal_x) || !std::isfinite(first_frame.focal_y) || first_frame.focal_x <= 0.0f || first_frame.focal_y <= 0.0f) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid focal length.", frame_set.name)};
                if (!std::isfinite(first_frame.principal_x) || !std::isfinite(first_frame.principal_y) || first_frame.principal_x < 0.0f || first_frame.principal_y < 0.0f || first_frame.principal_x >= static_cast<float>(first_frame.width) || first_frame.principal_y >= static_cast<float>(first_frame.height)) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid principal point.", frame_set.name)};

                std::vector<std::uint8_t> pixels;
                std::vector<float> camera;
                for (const FrameView& frame : frame_set.frames) {
                    if (frame.width != first_frame.width || frame.height != first_frame.height) throw std::runtime_error{std::format("frame set '{}' contains mixed image dimensions.", frame_set.name)};
                    if (frame.focal_x != first_frame.focal_x || frame.focal_y != first_frame.focal_y || frame.principal_x != first_frame.principal_x || frame.principal_y != first_frame.principal_y) throw std::runtime_error{std::format("frame set '{}' contains mixed camera intrinsics.", frame_set.name)};
                    if (frame.camera.size() != config::camera_value_count) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} camera values; expected {}.", frame_set.name, frame.camera.size(), config::camera_value_count)};
                    if (frame.height > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(frame.width) / config::rgba_channel_count) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    const std::uint64_t rgba_byte_count = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * config::rgba_channel_count;
                    if (rgba_byte_count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    if (frame.rgba.size() != static_cast<std::size_t>(rgba_byte_count)) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} RGBA bytes; expected {}.", frame_set.name, frame.rgba.size(), rgba_byte_count)};
                    pixels.append_range(frame.rgba);
                    camera.append_range(frame.camera);
                }

                this->device.frame_sets.push_back({});
                DeviceFrameSet& device_frame_set = this->device.frame_sets.back();
                cuda::upload_dataset(pixels.data(), pixels.size(), camera.data(), camera.size(), device_frame_set.pixels, device_frame_set.camera);
                this->host.frame_sets.push_back(HostFrameSet{
                    .name = std::string{frame_set.name},
                    .frame_count = static_cast<std::uint32_t>(frame_set.frames.size()),
                    .width = first_frame.width,
                    .height = first_frame.height,
                    .focal_x = first_frame.focal_x,
                    .focal_y = first_frame.focal_y,
                    .principal_x = first_frame.principal_x,
                    .principal_y = first_frame.principal_y,
                    .pixel_bytes = pixels.size(),
                    .camera_values = camera.size(),
                });
            }
        } catch (...) {
            for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_dataset(frame_set.pixels, frame_set.camera);
            throw;
        }
    }

    HyFluid::~HyFluid() noexcept {
        for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_dataset(frame_set.pixels, frame_set.camera);
    }

    std::expected<OptimizationStats, std::string> HyFluid::optimize(const OptimizationRequest request) {
        (void)request;
        return std::unexpected{std::string{"optimize is not implemented in the clean HyFluid training framework."}};
    }

    std::expected<EvaluationStats, std::string> HyFluid::evaluate(const EvaluationRequest request) const {
        (void)request;
        return std::unexpected{std::string{"evaluate is not implemented in the clean HyFluid training framework."}};
    }

    std::expected<void, std::string> HyFluid::export_weights(const std::filesystem::path& path) const {
        (void)path;
        return std::unexpected{std::string{"export_weights is not implemented in the clean HyFluid training framework."}};
    }

    std::expected<void, std::string> HyFluid::load_weights(const std::filesystem::path& path) {
        (void)path;
        return std::unexpected{std::string{"load_weights is not implemented in the clean HyFluid training framework."}};
    }
} // namespace hyfluid::train
