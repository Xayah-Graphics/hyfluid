module;
#include "hyfluid.train.h"

#include "hyfluid.train.config.h"

#include "json/json.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
module hyfluid.train;
import std;

namespace hyfluid::train {
    namespace {
        [[nodiscard]] std::array<float, 16u> swap_xz_axis(const std::span<const float> matrix) {
            if (matrix.size() != 16uz) throw std::runtime_error{"matrix must contain 16 values."};
            std::array<float, 16u> result = {};
            for (std::size_t row = 0uz; row < 4uz; ++row) {
                result[row * 4uz + 0uz] = matrix[row * 4uz + 2uz];
                result[row * 4uz + 1uz] = matrix[row * 4uz + 1uz];
                result[row * 4uz + 2uz] = matrix[row * 4uz + 0uz];
                result[row * 4uz + 3uz] = matrix[row * 4uz + 3uz];
            }
            for (const float value : result)
                if (!std::isfinite(value)) throw std::runtime_error{"matrix contains non-finite values."};
            return result;
        }

        [[nodiscard]] std::array<float, 16u> invert_matrix_4x4(const std::span<const float> matrix) {
            if (matrix.size() != 16uz) throw std::runtime_error{"matrix must contain 16 values."};
            std::array<float, 32u> inverse_work = {};
            for (std::size_t row = 0uz; row < 4uz; ++row) {
                for (std::size_t column = 0uz; column < 4uz; ++column) inverse_work[row * 8uz + column] = matrix[row * 4uz + column];
                inverse_work[row * 8uz + 4uz + row] = 1.0f;
            }
            for (std::size_t pivot_column = 0uz; pivot_column < 4uz; ++pivot_column) {
                std::size_t pivot_row = pivot_column;
                float pivot_abs       = std::fabs(inverse_work[pivot_row * 8uz + pivot_column]);
                for (std::size_t row = pivot_column + 1uz; row < 4uz; ++row) {
                    const float candidate_abs = std::fabs(inverse_work[row * 8uz + pivot_column]);
                    if (candidate_abs > pivot_abs) {
                        pivot_abs = candidate_abs;
                        pivot_row = row;
                    }
                }
                if (pivot_abs <= 1.0e-12f) throw std::runtime_error{"matrix is singular."};
                if (pivot_row != pivot_column)
                    for (std::size_t column = 0uz; column < 8uz; ++column) std::swap(inverse_work[pivot_row * 8uz + column], inverse_work[pivot_column * 8uz + column]);
                const float pivot = inverse_work[pivot_column * 8uz + pivot_column];
                for (std::size_t column = 0uz; column < 8uz; ++column) inverse_work[pivot_column * 8uz + column] /= pivot;
                for (std::size_t row = 0uz; row < 4uz; ++row) {
                    if (row == pivot_column) continue;
                    const float factor = inverse_work[row * 8uz + pivot_column];
                    for (std::size_t column = 0uz; column < 8uz; ++column) inverse_work[row * 8uz + column] -= factor * inverse_work[pivot_column * 8uz + column];
                }
            }
            std::array<float, 16u> inverse = {};
            for (std::size_t row = 0uz; row < 4uz; ++row)
                for (std::size_t column = 0uz; column < 4uz; ++column) inverse[row * 4uz + column] = inverse_work[row * 8uz + 4uz + column];
            for (const float value : inverse)
                if (!std::isfinite(value)) throw std::runtime_error{"matrix inverse contains non-finite values."};
            return inverse;
        }

        [[nodiscard]] std::array<float, 12u> unit_aabb_camera_from_transform_matrix(const std::span<const float> transform_matrix, const std::array<float, 16u>& world_to_sim, const std::array<float, 3u>& voxel_scale) {
            if (transform_matrix.size() != 16uz) throw std::runtime_error{"transform_matrix must contain 16 values."};
            for (const float value : transform_matrix)
                if (!std::isfinite(value)) throw std::runtime_error{"transform_matrix contains non-finite values."};

            const std::array raw_origin                          = {transform_matrix[3uz], transform_matrix[7uz], transform_matrix[11uz]};
            const std::array<std::array<float, 3u>, 3u> raw_axes = {{
                {transform_matrix[0uz], transform_matrix[4uz], transform_matrix[8uz]},
                {-transform_matrix[1uz], -transform_matrix[5uz], -transform_matrix[9uz]},
                {-transform_matrix[2uz], -transform_matrix[6uz], -transform_matrix[10uz]},
            }};

            std::array<float, 12u> camera = {};
            camera[0uz]                   = (world_to_sim[0uz] * raw_axes[0uz][0uz] + world_to_sim[1uz] * raw_axes[0uz][1uz] + world_to_sim[2uz] * raw_axes[0uz][2uz]) / voxel_scale[0uz];
            camera[1uz]                   = (world_to_sim[4uz] * raw_axes[0uz][0uz] + world_to_sim[5uz] * raw_axes[0uz][1uz] + world_to_sim[6uz] * raw_axes[0uz][2uz]) / voxel_scale[1uz];
            camera[2uz]                   = (world_to_sim[8uz] * raw_axes[0uz][0uz] + world_to_sim[9uz] * raw_axes[0uz][1uz] + world_to_sim[10uz] * raw_axes[0uz][2uz]) / voxel_scale[2uz];
            camera[3uz]                   = (world_to_sim[0uz] * raw_axes[1uz][0uz] + world_to_sim[1uz] * raw_axes[1uz][1uz] + world_to_sim[2uz] * raw_axes[1uz][2uz]) / voxel_scale[0uz];
            camera[4uz]                   = (world_to_sim[4uz] * raw_axes[1uz][0uz] + world_to_sim[5uz] * raw_axes[1uz][1uz] + world_to_sim[6uz] * raw_axes[1uz][2uz]) / voxel_scale[1uz];
            camera[5uz]                   = (world_to_sim[8uz] * raw_axes[1uz][0uz] + world_to_sim[9uz] * raw_axes[1uz][1uz] + world_to_sim[10uz] * raw_axes[1uz][2uz]) / voxel_scale[2uz];
            camera[6uz]                   = (world_to_sim[0uz] * raw_axes[2uz][0uz] + world_to_sim[1uz] * raw_axes[2uz][1uz] + world_to_sim[2uz] * raw_axes[2uz][2uz]) / voxel_scale[0uz];
            camera[7uz]                   = (world_to_sim[4uz] * raw_axes[2uz][0uz] + world_to_sim[5uz] * raw_axes[2uz][1uz] + world_to_sim[6uz] * raw_axes[2uz][2uz]) / voxel_scale[1uz];
            camera[8uz]                   = (world_to_sim[8uz] * raw_axes[2uz][0uz] + world_to_sim[9uz] * raw_axes[2uz][1uz] + world_to_sim[10uz] * raw_axes[2uz][2uz]) / voxel_scale[2uz];
            camera[9uz]                   = (world_to_sim[0uz] * raw_origin[0uz] + world_to_sim[1uz] * raw_origin[1uz] + world_to_sim[2uz] * raw_origin[2uz] + world_to_sim[3uz]) / voxel_scale[0uz];
            camera[10uz]                  = (world_to_sim[4uz] * raw_origin[0uz] + world_to_sim[5uz] * raw_origin[1uz] + world_to_sim[6uz] * raw_origin[2uz] + world_to_sim[7uz]) / voxel_scale[1uz];
            camera[11uz]                  = (world_to_sim[8uz] * raw_origin[0uz] + world_to_sim[9uz] * raw_origin[1uz] + world_to_sim[10uz] * raw_origin[2uz] + world_to_sim[11uz]) / voxel_scale[2uz];
            for (const float value : camera)
                if (!std::isfinite(value)) throw std::runtime_error{"transform_matrix maps to non-finite unit AABB camera values."};
            return camera;
        }

        struct SafetensorsTensor final {
            std::string_view name;
            std::uint32_t param_offset;
            std::uint64_t rows;
            std::uint64_t cols;
        };

        constexpr std::array density_weight_tensors = std::to_array<SafetensorsTensor>({
            SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = config::network_parameter_layout.mlp_input_weight_offset, .rows = config::mlp_width, .cols = config::mlp_input_width},
            SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = config::network_parameter_layout.mlp_output_weight_offset, .rows = config::network_output_width, .cols = config::mlp_width},
            SafetensorsTensor{.name = "density_global_rgb.param", .param_offset = config::network_parameter_layout.global_rgb_offset, .rows = 1u, .cols = 1u},
            SafetensorsTensor{.name = "hash4.params", .param_offset = config::network_parameter_layout.hash4_param_offset, .rows = config::network_parameter_layout.hash4_param_count / config::hash4_features_per_level, .cols = config::hash4_features_per_level},
        });

        [[nodiscard]] std::string density_hash4_resolutions_text() {
            std::string text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count; ++i) {
                if (!text.empty()) text += ",";
                text += std::format("{}", config::hash4_resolutions[i]);
            }
            return text;
        }

        [[nodiscard]] std::string density_hash4_offsets_text() {
            std::string text;
            for (std::uint32_t i = 0u; i < config::hash4_level_count + 1u; ++i) {
                if (!text.empty()) text += ",";
                text += std::format("{}", config::network_parameter_layout.hash4_offsets[i]);
            }
            return text;
        }

        [[nodiscard]] nlohmann::json density_weights_metadata() {
            const std::string hash4_resolutions_text = density_hash4_resolutions_text();
            const std::string hash4_offsets_text     = density_hash4_offsets_text();
            nlohmann::json metadata                  = nlohmann::json::object();
            metadata["format"]                       = "hyfluid.density.weights.v1";
            metadata["stage"]                        = "density";
            metadata["architecture_fingerprint"]     = std::format("hash4:l{}:f{}:max{}:res{}:offsets{}|mlp:w{}:in{}:out{}:global{}:total{}", config::hash4_level_count, config::hash4_features_per_level, config::hash4_max_entries, hash4_resolutions_text, hash4_offsets_text, config::mlp_width, config::mlp_input_width, config::network_output_width, config::network_parameter_layout.global_rgb_offset, config::network_parameter_layout.total_param_count);
            metadata["hash4_level_count"]            = std::format("{}", config::hash4_level_count);
            metadata["hash4_features_per_level"]     = std::format("{}", config::hash4_features_per_level);
            metadata["hash4_max_entries"]            = std::format("{}", config::hash4_max_entries);
            metadata["hash4_resolutions"]            = hash4_resolutions_text;
            metadata["hash4_offsets"]                = hash4_offsets_text;
            metadata["mlp_width"]                    = std::format("{}", config::mlp_width);
            metadata["mlp_input_width"]              = std::format("{}", config::mlp_input_width);
            metadata["network_output_width"]         = std::format("{}", config::network_output_width);
            metadata["global_rgb_offset"]            = std::format("{}", config::network_parameter_layout.global_rgb_offset);
            metadata["total_param_count"]            = std::format("{}", config::network_parameter_layout.total_param_count);
            metadata["sample_coord_floats"]          = std::format("{}", config::sample_coord_floats);
            return metadata;
        }

        [[nodiscard]] std::optional<std::size_t> find_frame_set_index(const std::span<const HyFluid::HostFrameSet> frame_sets, const std::string_view name) {
            for (std::size_t i = 0uz; i < frame_sets.size(); ++i)
                if (frame_sets[i].name == name) return i;
            return std::nullopt;
        }
    } // namespace

    void HyFluid::initialize(const std::span<const FrameSetView> frame_sets, const std::span<const float> voxel_matrix, const std::span<const float> voxel_scale) {
        try {
            if (frame_sets.empty()) throw std::runtime_error{"dynamic dataset must contain at least one frame set."};
            if (voxel_scale.size() != 3uz) throw std::runtime_error{"voxel_scale must contain 3 values."};
            std::array<float, 3u> field_voxel_scale = {};
            for (std::size_t i = 0uz; i < 3uz; ++i) {
                field_voxel_scale[i] = voxel_scale[i];
                if (!std::isfinite(field_voxel_scale[i]) || field_voxel_scale[i] == 0.0f) throw std::runtime_error{"voxel_scale contains invalid values."};
            }
            const std::array sim_to_world = swap_xz_axis(voxel_matrix);
            const std::array world_to_sim = invert_matrix_4x4(std::span{sim_to_world.data(), sim_to_world.size()});
            std::array<float, 9u> field_to_world_linear = {};
            for (std::size_t row = 0uz; row < 3uz; ++row)
                for (std::size_t column = 0uz; column < 3uz; ++column) field_to_world_linear[row * 3uz + column] = sim_to_world[row * 4uz + column] * field_voxel_scale[column];
            for (std::size_t column = 0uz; column < 3uz; ++column) {
                const float x      = field_to_world_linear[column];
                const float y      = field_to_world_linear[3uz + column];
                const float z      = field_to_world_linear[6uz + column];
                const float extent = std::sqrt(x * x + y * y + z * z);
                if (!std::isfinite(extent) || extent <= 0.0f) throw std::runtime_error{"field_to_world_linear contains a degenerate axis."};
            }

            // ====================================================================================================
            // 1. UPLOAD FRAME SETS TO GPU
            // ====================================================================================================
            {
                this->host.frame_sets.reserve(frame_sets.size());
                this->device.frame_sets.reserve(frame_sets.size());
                for (const FrameSetView& frame_set : frame_sets) {
                    if (frame_set.name.empty()) throw std::runtime_error{"frame set name must not be empty."};
                    if (frame_set.view_count == 0u || frame_set.time_count == 0u) throw std::runtime_error{std::format("frame set '{}' must declare positive view_count and time_count.", frame_set.name)};
                    if (frame_set.frames.empty()) throw std::runtime_error{std::format("frame set '{}' contains no frames.", frame_set.name)};
                    if (frame_set.view_count > std::numeric_limits<std::uint32_t>::max() / frame_set.time_count) throw std::runtime_error{std::format("frame set '{}' view-time grid is too large.", frame_set.name)};
                    const std::uint32_t expected_frame_count = frame_set.view_count * frame_set.time_count;
                    if (frame_set.frames.size() != expected_frame_count) throw std::runtime_error{std::format("frame set '{}' contains {} frames; expected dense {}x{} grid.", frame_set.name, frame_set.frames.size(), frame_set.view_count, frame_set.time_count)};
                    for (const HostFrameSet& existing_frame_set : this->host.frame_sets)
                        if (existing_frame_set.name == frame_set.name) throw std::runtime_error{std::format("frame set '{}' was loaded more than once.", frame_set.name)};
                    if (this->host.frame_sets.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error{"dynamic dataset contains too many frame sets."};

                    const FrameView& first_frame = frame_set.frames.front();
                    if (first_frame.width == 0u || first_frame.height == 0u) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid dimensions.", frame_set.name)};
                    if (first_frame.height > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(first_frame.width) / 4ull) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    const std::uint64_t frame_pixel_bytes = static_cast<std::uint64_t>(first_frame.width) * static_cast<std::uint64_t>(first_frame.height) * 4ull;
                    if (frame_pixel_bytes > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                    if (frame_set.frames.size() > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(frame_pixel_bytes)) throw std::runtime_error{std::format("frame set '{}' pixel buffer is too large.", frame_set.name)};

                    std::vector<std::uint8_t> pixels;
                    std::vector<float> camera;
                    std::vector<float> intrinsics;
                    std::vector frame_indices(frame_set.frames.size(), std::numeric_limits<std::uint32_t>::max());
                    std::vector<std::array<float, 12u>> view_camera(frame_set.view_count);
                    std::vector<std::array<float, 4u>> view_intrinsics(frame_set.view_count);
                    std::vector<std::uint8_t> view_reference_seen(frame_set.view_count, 0u);
                    pixels.reserve(frame_set.frames.size() * static_cast<std::size_t>(frame_pixel_bytes));
                    camera.reserve(frame_set.frames.size() * 12uz);
                    intrinsics.reserve(frame_set.frames.size() * 4uz);

                    std::uint32_t frame_index = 0u;
                    for (const FrameView& frame : frame_set.frames) {
                        if (frame.width != first_frame.width || frame.height != first_frame.height) throw std::runtime_error{std::format("frame set '{}' contains mixed image dimensions.", frame_set.name)};
                        if (frame.rgba.size() != static_cast<std::size_t>(frame_pixel_bytes)) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} RGBA bytes; expected {}.", frame_set.name, frame.rgba.size(), frame_pixel_bytes)};
                        if (!std::isfinite(frame.focal_x) || !std::isfinite(frame.focal_y) || frame.focal_x <= 0.0f || frame.focal_y <= 0.0f) throw std::runtime_error{std::format("frame set '{}' contains invalid focal length.", frame_set.name)};
                        if (!std::isfinite(frame.principal_x) || !std::isfinite(frame.principal_y) || frame.principal_x < 0.0f || frame.principal_y < 0.0f || frame.principal_x >= static_cast<float>(frame.width) || frame.principal_y >= static_cast<float>(frame.height)) throw std::runtime_error{std::format("frame set '{}' contains invalid principal point.", frame_set.name)};
                        if (frame.view_index >= frame_set.view_count) throw std::runtime_error{std::format("frame set '{}' contains view_index {} outside view_count {}.", frame_set.name, frame.view_index, frame_set.view_count)};
                        if (frame.time_index >= frame_set.time_count) throw std::runtime_error{std::format("frame set '{}' contains time_index {} outside time_count {}.", frame_set.name, frame.time_index, frame_set.time_count)};
                        const std::array frame_camera     = unit_aabb_camera_from_transform_matrix(frame.transform_matrix, world_to_sim, field_voxel_scale);
                        const std::array frame_intrinsics = {frame.focal_x, frame.focal_y, frame.principal_x, frame.principal_y};
                        if (view_reference_seen[frame.view_index] == 0u) {
                            for (std::size_t i = 0uz; i < 12uz; ++i) view_camera[frame.view_index][i] = frame_camera[i];
                            view_intrinsics[frame.view_index]     = frame_intrinsics;
                            view_reference_seen[frame.view_index] = 1u;
                        } else {
                            for (std::size_t i = 0uz; i < 12uz; ++i)
                                if (view_camera[frame.view_index][i] != frame_camera[i]) throw std::runtime_error{std::format("frame set '{}' contains changing camera values for view {}.", frame_set.name, frame.view_index)};
                            for (std::size_t i = 0uz; i < 4uz; ++i)
                                if (view_intrinsics[frame.view_index][i] != frame_intrinsics[i]) throw std::runtime_error{std::format("frame set '{}' contains changing intrinsics for view {}.", frame_set.name, frame.view_index)};
                        }

                        const std::uint32_t frame_grid_index = frame.view_index * frame_set.time_count + frame.time_index;
                        if (frame_indices[frame_grid_index] != std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains duplicate frame at view {} time {}.", frame_set.name, frame.view_index, frame.time_index)};
                        frame_indices[frame_grid_index] = frame_index;

                        pixels.append_range(frame.rgba);
                        camera.append_range(frame_camera);
                        intrinsics.push_back(frame.focal_x);
                        intrinsics.push_back(frame.focal_y);
                        intrinsics.push_back(frame.principal_x);
                        intrinsics.push_back(frame.principal_y);
                        ++frame_index;
                    }
                    for (const std::uint32_t stored_frame_index : frame_indices)
                        if (stored_frame_index == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("frame set '{}' is missing at least one view-time frame.", frame_set.name)};

                    this->device.frame_sets.push_back({});
                    DeviceFrameSet& device_frame_set = this->device.frame_sets.back();
                    cuda::upload_dataset(pixels.data(), pixels.size(), camera.data(), camera.size(), intrinsics.data(), intrinsics.size(), frame_indices.data(), frame_indices.size(), device_frame_set.pixels, device_frame_set.camera, device_frame_set.intrinsics, device_frame_set.frame_indices);

                    this->host.frame_sets.push_back(HostFrameSet{
                        .name       = std::string{frame_set.name},
                        .view_count = frame_set.view_count,
                        .time_count = frame_set.time_count,
                        .width      = first_frame.width,
                        .height     = first_frame.height,
                    });
                }
            }

            std::uint64_t evaluation_pixel_capacity = 0ull;
            for (const HostFrameSet& frame_set : this->host.frame_sets) {
                const std::uint64_t render_pixels = static_cast<std::uint64_t>(frame_set.width) * frame_set.height;
                evaluation_pixel_capacity         = std::max(evaluation_pixel_capacity, render_pixels);
            }
            if (evaluation_pixel_capacity == 0ull || evaluation_pixel_capacity > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"evaluation render pixel capacity is invalid."};
            this->host.evaluation_pixel_capacity = static_cast<std::uint32_t>(evaluation_pixel_capacity);

            cuda::upload_field_domain(field_to_world_linear.data(), this->device.field_to_world_linear);
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
            for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.frame_indices);
            throw;
        }
    }

    HyFluid::~HyFluid() noexcept {
        cuda::destroy_network_handle(this->device.cublaslt_handle);
        cuda::free_device_buffers(this->device.field_to_world_linear, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.occupancy_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.network_output_gradients, this->device.network_input, this->device.network_hidden, this->device.network_output, this->device.network_input_gradients, this->device.network_hidden_gradients, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.evaluation_pixels);
        for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera, frame_set.intrinsics, frame_set.frame_indices);
    }

    std::expected<OptimizationStats, std::string> HyFluid::optimize(const OptimizationRequest& request) {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"optimization frame set must not be empty."};
            if (request.iterations < 1) throw std::runtime_error{"optimization iterations must be positive."};
            const std::optional<std::size_t> frame_set_index = find_frame_set_index(this->host.frame_sets, request.frame_set);
            if (!frame_set_index.has_value()) throw std::runtime_error{std::format("optimization frame set '{}' is not loaded.", request.frame_set)};
            const HostFrameSet* const host_frame_set     = std::addressof(this->host.frame_sets[*frame_set_index]);
            const DeviceFrameSet* const device_frame_set = std::addressof(this->device.frame_sets[*frame_set_index]);

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

            const std::optional<std::size_t> frame_set_index = find_frame_set_index(this->host.frame_sets, request.frame_set);
            if (!frame_set_index.has_value()) throw std::runtime_error{std::format("evaluation frame set '{}' is not loaded.", request.frame_set)};
            const HostFrameSet* const host_frame_set     = std::addressof(this->host.frame_sets[*frame_set_index]);
            const DeviceFrameSet* const device_frame_set = std::addressof(this->device.frame_sets[*frame_set_index]);
            if (host_frame_set->view_count == 0u || host_frame_set->time_count == 0u) throw std::runtime_error{std::format("evaluation frame set '{}' is not a dense view-time grid.", host_frame_set->name)};

            const std::uint32_t render_width  = host_frame_set->width;
            const std::uint32_t render_height = host_frame_set->height;
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

            std::vector<float> host_params(config::network_parameter_layout.total_param_count);
            cuda::download_trainable_parameters(this->device.params_full_precision, host_params.data());

            nlohmann::json header  = nlohmann::json::object();
            header["__metadata__"] = density_weights_metadata();

            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : density_weight_tensors) {
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
            for (const SafetensorsTensor& tensor : density_weight_tensors) output.write(reinterpret_cast<const char*>(host_params.data() + tensor.param_offset), static_cast<std::streamsize>(tensor.rows * tensor.cols * sizeof(float)));
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

            const nlohmann::json expected_metadata = density_weights_metadata();

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
            if (header.size() != density_weight_tensors.size() + 1uz) throw std::runtime_error{"safetensors header contains unexpected tensors."};
            if (!header.contains("__metadata__") || !header.at("__metadata__").is_object()) throw std::runtime_error{"safetensors metadata is missing."};
            if (header.at("__metadata__") != expected_metadata) throw std::runtime_error{"safetensors metadata does not match the current HyFluid density configuration."};

            std::uint64_t expected_data_offset = 0u;
            for (const SafetensorsTensor& tensor : density_weight_tensors) {
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
            for (const SafetensorsTensor& tensor : density_weight_tensors) {
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
