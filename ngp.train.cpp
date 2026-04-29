module;
#include "ngp.train.h"

#include "json/json.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
module hyfluid.train;
import std;
import hyfluid.dataset;

namespace hyfluid::train {
    HyFluidDensity::HyFluidDensity(const hyfluid::dataset::ScalarRealDataset& dataset, const TrainOptions& options) {
        try {
            if (dataset.train.empty()) throw std::runtime_error{"ScalarReal dataset has no training videos."};
            if (dataset.test.empty()) throw std::runtime_error{"ScalarReal dataset has no test videos."};
            if (options.rays_per_step == 0u) throw std::runtime_error{"rays_per_step must be positive."};
            if (options.samples_per_ray == 0u || options.samples_per_ray > 512u) throw std::runtime_error{"samples_per_ray must be in [1,512]."};
            if (options.test_frame_limit == 0u) throw std::runtime_error{"test_frame_limit must be positive."};
            if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0f) throw std::runtime_error{"learning_rate must be positive."};

            const hyfluid::dataset::ScalarRealVideo& first_train = dataset.train.front();
            if (first_train.width == 0u || first_train.height == 0u || first_train.frame_count == 0u) throw std::runtime_error{"invalid ScalarReal training video metadata."};
            this->host.options          = options;
            this->host.train_view_count = static_cast<std::uint32_t>(dataset.train.size());
            this->host.test_view_count  = static_cast<std::uint32_t>(dataset.test.size());
            this->host.frame_count      = first_train.frame_count;
            this->host.width            = first_train.width;
            this->host.height           = first_train.height;
            this->host.near_plane       = dataset.near;
            this->host.far_plane        = dataset.far;
            this->host.sim_to_world     = dataset.sim_to_world;
            this->host.world_to_sim     = dataset.world_to_sim;
            this->host.voxel_scale      = dataset.voxel_scale;

            std::vector<std::uint8_t> train_pixels;
            std::vector<float> train_cameras;
            std::uint64_t train_byte_count = 0ull;
            for (const hyfluid::dataset::ScalarRealVideo& video : dataset.train) {
                if (video.width != this->host.width || video.height != this->host.height || video.frame_count != this->host.frame_count) throw std::runtime_error{"training videos must share resolution and frame count."};
                train_byte_count += static_cast<std::uint64_t>(video.rgb.size());
            }
            if (train_byte_count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{"training videos are too large for this host."};
            train_pixels.reserve(static_cast<std::size_t>(train_byte_count));
            train_cameras.reserve(static_cast<std::size_t>(this->host.train_view_count) * 17);
            for (const hyfluid::dataset::ScalarRealVideo& video : dataset.train) {
                if (video.rgb.size() != static_cast<std::size_t>(video.frame_count) * video.width * video.height * 3) throw std::runtime_error{std::format("{} has invalid RGB byte count.", video.file_name)};
                train_pixels.append_range(video.rgb);
                train_cameras.append_range(video.camera);
                train_cameras.push_back(video.focal);
            }

            std::vector<std::uint8_t> test_pixels;
            std::vector<float> test_cameras;
            std::uint64_t test_byte_count = 0ull;
            for (const hyfluid::dataset::ScalarRealVideo& video : dataset.test) {
                if (video.width != this->host.width || video.height != this->host.height || video.frame_count != this->host.frame_count) throw std::runtime_error{"test videos must match training resolution and frame count."};
                test_byte_count += static_cast<std::uint64_t>(video.rgb.size());
            }
            if (test_byte_count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{"test videos are too large for this host."};
            test_pixels.reserve(static_cast<std::size_t>(test_byte_count));
            test_cameras.reserve(static_cast<std::size_t>(this->host.test_view_count) * 17);
            for (const hyfluid::dataset::ScalarRealVideo& video : dataset.test) {
                if (video.rgb.size() != static_cast<std::size_t>(video.frame_count) * video.width * video.height * 3) throw std::runtime_error{std::format("{} has invalid RGB byte count.", video.file_name)};
                test_pixels.append_range(video.rgb);
                test_cameras.append_range(video.camera);
                test_cameras.push_back(video.focal);
            }

            constexpr std::uint32_t min_resolution = 16u;
            constexpr std::uint32_t max_resolution = 512u;
            constexpr std::uint32_t max_entries    = 1u << 19u;
            const double scale_base                = std::exp((std::log(static_cast<double>(max_resolution)) - std::log(static_cast<double>(min_resolution))) / static_cast<double>(hyfluid::cuda::HASH_LEVELS - 1u));
            std::uint32_t hash_offset              = 0u;
            for (std::uint32_t level = 0u; level < hyfluid::cuda::HASH_LEVELS; ++level) {
                const std::uint32_t resolution     = static_cast<std::uint32_t>(std::ceil(static_cast<double>(min_resolution) * std::pow(scale_base, static_cast<double>(level))));
                const std::uint64_t raw_entries    = static_cast<std::uint64_t>(resolution + 1u) * static_cast<std::uint64_t>(resolution + 1u) * static_cast<std::uint64_t>(resolution + 1u) * static_cast<std::uint64_t>(resolution + 1u);
                const std::uint64_t padded_entries = ((raw_entries + 7ull) / 8ull) * 8ull;
                const std::uint32_t entries        = static_cast<std::uint32_t>(std::min<std::uint64_t>(padded_entries, max_entries));
                this->host.hash_offsets[level]     = hash_offset;
                this->host.hash_entries[level]     = entries;
                this->host.hash_dense[level]       = raw_entries <= entries ? 1u : 0u;
                for (std::uint32_t axis = 0u; axis < 4u; ++axis) this->host.hash_resolutions[level * 4u + axis] = resolution;
                hash_offset += entries * hyfluid::cuda::HASH_FEATURES_PER_LEVEL;
            }

            this->host.hash_param_count = hash_offset;
            this->host.w1_offset        = this->host.hash_param_count;
            this->host.w2_offset        = this->host.w1_offset + hyfluid::cuda::MLP_HIDDEN_WIDTH * hyfluid::cuda::HASH_INPUT_WIDTH;
            this->host.color_offset     = this->host.w2_offset + hyfluid::cuda::MLP_HIDDEN_WIDTH;
            this->host.param_count      = this->host.color_offset + 1u;

            hyfluid::cuda::upload_bytes(train_pixels.data(), train_pixels.size(), this->device.train_pixels);
            hyfluid::cuda::upload_bytes(test_pixels.data(), test_pixels.size(), this->device.test_pixels);
            hyfluid::cuda::upload_floats(train_cameras.data(), train_cameras.size(), this->device.train_cameras);
            hyfluid::cuda::upload_floats(test_cameras.data(), test_cameras.size(), this->device.test_cameras);
            hyfluid::cuda::upload_floats(this->host.sim_to_world.data(), this->host.sim_to_world.size(), this->device.sim_to_world);
            hyfluid::cuda::upload_floats(this->host.world_to_sim.data(), this->host.world_to_sim.size(), this->device.world_to_sim);
            hyfluid::cuda::upload_floats(this->host.voxel_scale.data(), this->host.voxel_scale.size(), this->device.voxel_scale);
            hyfluid::cuda::upload_uint32s(this->host.hash_offsets.data(), this->host.hash_offsets.size(), this->device.hash_offsets);
            hyfluid::cuda::upload_uint32s(this->host.hash_entries.data(), this->host.hash_entries.size(), this->device.hash_entries);
            hyfluid::cuda::upload_uint32s(this->host.hash_resolutions.data(), this->host.hash_resolutions.size(), this->device.hash_resolutions);
            hyfluid::cuda::upload_uint32s(this->host.hash_dense.data(), this->host.hash_dense.size(), this->device.hash_dense);
            hyfluid::cuda::allocate_float_buffer(this->host.param_count, this->device.params);
            hyfluid::cuda::allocate_float_buffer(this->host.param_count, this->device.gradients);
            hyfluid::cuda::allocate_float_buffer(this->host.param_count, this->device.first_moments);
            hyfluid::cuda::allocate_float_buffer(this->host.param_count, this->device.second_moments);
            hyfluid::cuda::allocate_float_buffer(this->host.options.rays_per_step, this->device.loss_values);
            hyfluid::cuda::allocate_double_buffer(1, this->device.test_loss_sum);
            hyfluid::cuda::allocate_byte_buffer(static_cast<std::size_t>(this->host.width) * this->host.height * 2 * 3, this->device.comparison_pixels);
            hyfluid::cuda::allocate_byte_buffer(static_cast<std::size_t>(hyfluid::cuda::TIME_OCCUPANCY_BINS) * hyfluid::cuda::OCCUPANCY_GRID_CELLS / 8, this->device.occupancy_bits);
            hyfluid::cuda::allocate_float_buffer(static_cast<std::size_t>(hyfluid::cuda::TIME_OCCUPANCY_BINS) * hyfluid::cuda::OCCUPANCY_GRID_CELLS, this->device.occupancy_values);
            hyfluid::cuda::allocate_uint32_buffer(hyfluid::cuda::TIME_OCCUPANCY_BINS, this->device.occupancy_counts);
            hyfluid::cuda::allocate_uint32_buffer(1, this->device.occupancy_skipped_samples);
            hyfluid::cuda::initialize_parameters(this->host.param_count, this->host.hash_param_count, this->host.w1_offset, this->host.w2_offset, this->host.color_offset, this->device.params, this->device.gradients, this->device.first_moments, this->device.second_moments);
            hyfluid::cuda::initialize_occupancy(this->device.occupancy_bits, this->device.occupancy_values, this->device.occupancy_counts);
        } catch (...) {
            this->~HyFluidDensity();
            throw;
        }
    }

    HyFluidDensity::~HyFluidDensity() noexcept {
        void* pointers[] = {
            const_cast<std::uint8_t*>(this->device.train_pixels),
            const_cast<std::uint8_t*>(this->device.test_pixels),
            const_cast<float*>(this->device.train_cameras),
            const_cast<float*>(this->device.test_cameras),
            const_cast<float*>(this->device.sim_to_world),
            const_cast<float*>(this->device.world_to_sim),
            const_cast<float*>(this->device.voxel_scale),
            const_cast<std::uint32_t*>(this->device.hash_offsets),
            const_cast<std::uint32_t*>(this->device.hash_entries),
            const_cast<std::uint32_t*>(this->device.hash_resolutions),
            const_cast<std::uint32_t*>(this->device.hash_dense),
            this->device.params,
            this->device.gradients,
            this->device.first_moments,
            this->device.second_moments,
            this->device.loss_values,
            this->device.test_loss_sum,
            this->device.comparison_pixels,
            this->device.occupancy_bits,
            this->device.occupancy_values,
            this->device.occupancy_counts,
            this->device.occupancy_skipped_samples,
        };
        hyfluid::cuda::free_device_buffers(pointers, std::size(pointers));
        this->device = {};
    }

    std::expected<TrainStats, std::string> HyFluidDensity::train(const std::int32_t iters) {
        try {
            if (iters <= 0) throw std::runtime_error{"train iteration count must be positive."};
            const auto start_time = std::chrono::steady_clock::now();
            for (std::int32_t i = 0; i < iters; ++i) {
                hyfluid::cuda::run_training_step(this->device.train_pixels, this->device.train_cameras, this->host.train_view_count, this->host.frame_count, this->host.width, this->host.height, this->device.world_to_sim, this->device.voxel_scale, this->host.near_plane, this->host.far_plane, this->host.options.samples_per_ray, this->host.options.rays_per_step, this->host.current_step, this->device.hash_offsets, this->device.hash_entries, this->device.hash_resolutions, this->device.hash_dense, this->host.hash_param_count, this->host.w1_offset, this->host.w2_offset, this->host.color_offset, this->device.occupancy_bits, this->device.params, this->device.gradients, this->device.loss_values, this->device.occupancy_skipped_samples);
                const float learning_rate = this->host.options.learning_rate * std::pow(0.1f, static_cast<float>(this->host.current_step) / 250.0f);
                hyfluid::cuda::step_radam(this->host.param_count, this->host.hash_param_count, this->host.current_step, learning_rate, this->device.params, this->device.gradients, this->device.first_moments, this->device.second_moments);
                const std::uint32_t occupancy_bin = this->host.current_step % hyfluid::cuda::TIME_OCCUPANCY_BINS;
                const auto occupancy_start_time   = std::chrono::steady_clock::now();
                hyfluid::cuda::update_time_occupancy(occupancy_bin, this->device.sim_to_world, this->device.voxel_scale, this->device.hash_offsets, this->device.hash_entries, this->device.hash_resolutions, this->device.hash_dense, this->host.w1_offset, this->host.w2_offset, this->device.params, this->device.occupancy_bits, this->device.occupancy_values, this->device.occupancy_counts);
                this->host.last_occupancy_update_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - occupancy_start_time).count();
                this->host.last_occupancy_bin       = occupancy_bin;
                hyfluid::cuda::download_uint32(this->device.occupancy_counts + occupancy_bin, this->host.last_occupancy_occupied_cells);
                hyfluid::cuda::download_uint32(this->device.occupancy_skipped_samples, this->host.last_occupancy_skipped_samples);
                ++this->host.current_step;
            }

            std::vector<float> loss_values(this->host.options.rays_per_step);
            hyfluid::cuda::download_floats(this->device.loss_values, loss_values.size(), loss_values.data());
            double loss_sum = 0.0;
            for (const float loss : loss_values) loss_sum += static_cast<double>(loss);
            return TrainStats{
                .step                      = this->host.current_step,
                .rays_per_step             = this->host.options.rays_per_step,
                .samples_per_ray           = this->host.options.samples_per_ray,
                .occupancy_bin             = this->host.last_occupancy_bin,
                .occupancy_occupied_cells  = this->host.last_occupancy_occupied_cells,
                .occupancy_skipped_samples = this->host.last_occupancy_skipped_samples,
                .loss                      = static_cast<float>(loss_sum / static_cast<double>(loss_values.size())),
                .occupancy_ratio           = static_cast<float>(static_cast<double>(this->host.last_occupancy_occupied_cells) / static_cast<double>(hyfluid::cuda::OCCUPANCY_GRID_CELLS)),
                .occupancy_update_ms       = this->host.last_occupancy_update_ms,
                .elapsed_ms                = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count(),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<TestStats, std::string> HyFluidDensity::test() const {
        try {
            if (this->host.options.test_output_dir.empty()) throw std::runtime_error{"test output directory must not be empty."};
            if (std::filesystem::exists(this->host.options.test_output_dir) && !std::filesystem::is_directory(this->host.options.test_output_dir)) throw std::runtime_error{std::format("test output path '{}' is not a directory.", this->host.options.test_output_dir.string())};
            std::filesystem::create_directories(this->host.options.test_output_dir);
            const auto start_time              = std::chrono::steady_clock::now();
            const std::uint32_t frames_to_test = std::min(this->host.options.test_frame_limit, this->host.frame_count);
            std::vector<std::uint8_t> comparison_pixels(static_cast<std::size_t>(this->host.width) * this->host.height * 2 * 3);
            double total_loss_sum     = 0.0;
            std::uint32_t image_count = 0u;

            for (std::uint32_t view = 0u; view < this->host.test_view_count; ++view) {
                for (std::uint32_t frame = 0u; frame < frames_to_test; ++frame) {
                    hyfluid::cuda::evaluate_test_frame(this->device.test_pixels, this->device.test_cameras, view, frame, this->host.frame_count, this->host.width, this->host.height, this->device.world_to_sim, this->device.voxel_scale, this->host.near_plane, this->host.far_plane, this->host.options.samples_per_ray, this->device.hash_offsets, this->device.hash_entries, this->device.hash_resolutions, this->device.hash_dense, this->host.w1_offset, this->host.w2_offset, this->host.color_offset, this->device.occupancy_bits, this->device.params, this->device.test_loss_sum, this->device.comparison_pixels);
                    double image_loss = 0.0;
                    hyfluid::cuda::download_double(this->device.test_loss_sum, image_loss);
                    hyfluid::cuda::download_bytes(this->device.comparison_pixels, comparison_pixels.size(), comparison_pixels.data());
                    total_loss_sum += image_loss;

                    const std::filesystem::path output_path = this->host.options.test_output_dir / std::format("test_v{:02}_f{:04}.png", view, frame);
                    const std::string output_path_text      = output_path.string();
                    const int output_width                  = static_cast<int>(this->host.width * 2u);
                    const int output_height                 = static_cast<int>(this->host.height);
                    if (stbi_write_png(output_path_text.c_str(), output_width, output_height, 3, comparison_pixels.data(), output_width * 3) == 0) throw std::runtime_error{std::format("failed to write '{}'.", output_path_text)};
                    ++image_count;
                }
            }

            const std::uint64_t pixel_count = static_cast<std::uint64_t>(image_count) * this->host.width * this->host.height;
            const double mse                = total_loss_sum / (static_cast<double>(pixel_count) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"test produced non-finite MSE."};
            return TestStats{
                .step                   = this->host.current_step,
                .image_count            = image_count,
                .comparison_image_count = image_count,
                .pixel_count            = pixel_count,
                .mse                    = static_cast<float>(mse),
                .psnr                   = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity(),
                .elapsed_ms             = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count(),
                .output_dir             = this->host.options.test_output_dir,
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluidDensity::export_weights(const std::filesystem::path& path) const {
        try {
            if (path.empty()) throw std::runtime_error{"weights export path must not be empty."};
            if (!path.parent_path().empty() && !std::filesystem::is_directory(path.parent_path())) throw std::runtime_error{std::format("weights export parent directory '{}' does not exist.", path.parent_path().string())};
            std::vector<float> params(this->host.param_count);
            hyfluid::cuda::download_floats(this->device.params, params.size(), params.data());

            nlohmann::json header  = nlohmann::json::object();
            header["__metadata__"] = {
                {"format", "hyfluid-density.v1"},
                {"step", std::format("{}", this->host.current_step)},
                {"param_count", std::format("{}", this->host.param_count)},
                {"hash_param_count", std::format("{}", this->host.hash_param_count)},
                {"w1_offset", std::format("{}", this->host.w1_offset)},
                {"w2_offset", std::format("{}", this->host.w2_offset)},
                {"color_offset", std::format("{}", this->host.color_offset)},
            };
            header["params"] = {
                {"dtype", "F32"},
                {"shape", nlohmann::json::array({this->host.param_count})},
                {"data_offsets", nlohmann::json::array({0, static_cast<std::uint64_t>(this->host.param_count) * sizeof(float)})},
            };
            const std::string header_text   = header.dump();
            const std::uint64_t header_size = header_text.size();
            std::ofstream output{path, std::ios::binary | std::ios::trunc};
            if (!output) throw std::runtime_error{std::format("failed to open '{}'.", path.string())};
            output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
            output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            output.write(reinterpret_cast<const char*>(params.data()), static_cast<std::streamsize>(params.size() * sizeof(float)));
            if (!output) throw std::runtime_error{std::format("failed to write '{}'.", path.string())};
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> HyFluidDensity::load_weights(const std::filesystem::path& path) {
        try {
            if (this->host.current_step != 0u) throw std::runtime_error{"weights can only be loaded before training starts."};
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error{std::format("weights file '{}' does not exist.", path.string())};
            std::ifstream input{path, std::ios::binary};
            if (!input) throw std::runtime_error{std::format("failed to open '{}'.", path.string())};
            std::uint64_t header_size = 0u;
            input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
            if (!input || header_size == 0u || header_size > (1ull << 24ull)) throw std::runtime_error{"weights header size is invalid."};
            std::string header_text(header_size, '\0');
            input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            if (!input) throw std::runtime_error{"failed to read weights header."};
            const nlohmann::json header    = nlohmann::json::parse(header_text);
            const nlohmann::json& metadata = header.at("__metadata__");
            if (metadata.at("format").get<std::string>() != "hyfluid-density.v1") throw std::runtime_error{"weights format is not hyfluid-density.v1."};
            if (std::stoul(metadata.at("param_count").get<std::string>()) != this->host.param_count) throw std::runtime_error{"weights param_count does not match this run."};
            if (std::stoul(metadata.at("hash_param_count").get<std::string>()) != this->host.hash_param_count) throw std::runtime_error{"weights hash_param_count does not match this run."};
            if (std::stoul(metadata.at("w1_offset").get<std::string>()) != this->host.w1_offset) throw std::runtime_error{"weights w1_offset does not match this run."};
            if (std::stoul(metadata.at("w2_offset").get<std::string>()) != this->host.w2_offset) throw std::runtime_error{"weights w2_offset does not match this run."};
            if (std::stoul(metadata.at("color_offset").get<std::string>()) != this->host.color_offset) throw std::runtime_error{"weights color_offset does not match this run."};
            std::vector<float> params(this->host.param_count);
            input.read(reinterpret_cast<char*>(params.data()), static_cast<std::streamsize>(params.size() * sizeof(float)));
            if (!input) throw std::runtime_error{"failed to read weights parameters."};
            hyfluid::cuda::upload_parameters(this->host.param_count, params.data(), this->device.params, this->device.gradients, this->device.first_moments, this->device.second_moments);
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::train
