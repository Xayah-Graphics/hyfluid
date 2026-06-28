import std;
import xayah.util.xcli;
import dataset.scalar_real;
import hyfluid.train;

int main(const int argc, const char* const* const argv) {
    const std::span arguments{argv, static_cast<std::size_t>(argc)};
    constexpr std::string_view ansi_reset = "\x1b[0m";
    constexpr std::string_view ansi_red   = "\x1b[31m";

    std::filesystem::path dataset_path;
    std::vector<std::string> requested_frame_sets;
    std::int32_t optimize_iterations = 0;

    xayah::util::Command command = xayah::util::Command{"ScalarReal dynamic dataset loader for HyFluid."}
                                 | xayah::util::positional(
                                     xayah::util::PositionalSpec{
                                         .name         = "dataset-path",
                                         .description  = "ScalarReal dataset directory.",
                                         .show_default = false,
                                         .required     = true,
                                     },
                                     dataset_path, xayah::util::PathRule{.requirement = xayah::util::PathRequirement::existing_directory})
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "frame-set",
                                         .short_name   = 'f',
                                         .value_name   = "name",
                                         .description  = "Frame set to load. Repeat for multiple sets.",
                                         .default_text = "train",
                                     },
                                     requested_frame_sets)
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "optimize",
                                         .value_name   = "iterations",
                                         .description  = "Run sampler-only optimization smoke iterations on the first loaded frame set.",
                                         .default_text = "0",
                                     },
                                     optimize_iterations)
                                 | xayah::util::example("data/ScalarReal")
                                 | xayah::util::example("data/ScalarReal --frame-set train --frame-set test")
                                 | xayah::util::example("data/ScalarReal --frame-set train --optimize 1");

    const std::string usage = command.help(arguments);
    const auto cli_result   = command.parse(arguments);
    if (!cli_result) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, cli_result.error());
        std::println("{}", usage);
        return 2;
    }
    if (cli_result->help_requested) {
        std::println("{}", usage);
        return 0;
    }

    if (!command.option_provided("frame-set")) requested_frame_sets.push_back("train");
    if (command.option_provided("optimize") && optimize_iterations < 1) {
        std::println("{}error:{} --optimize must be positive when provided.", ansi_red, ansi_reset);
        return 2;
    }
    for (std::size_t frame_set_index = 0uz; frame_set_index < requested_frame_sets.size(); ++frame_set_index) {
        const std::string& frame_set_name = requested_frame_sets[frame_set_index];
        if (frame_set_name != "train" && frame_set_name != "test") {
            std::println("{}error:{} ScalarReal frame set '{}' is not available; supported frame sets: train, test.", ansi_red, ansi_reset, frame_set_name);
            return 2;
        }
        for (std::size_t previous_index = 0uz; previous_index < frame_set_index; ++previous_index) {
            if (requested_frame_sets[previous_index] == frame_set_name) {
                std::println("{}error:{} ScalarReal frame set '{}' was requested more than once.", ansi_red, ansi_reset, frame_set_name);
                return 2;
            }
        }
    }

    const std::expected<dataset::scalar_real::Dataset, std::string> loaded_dataset = dataset::scalar_real::load(dataset_path, dataset::scalar_real::LoadRequest{
                                                                                                                                  .frame_sets  = requested_frame_sets,
                                                                                                                                  .scene_scale = 1.0f,
                                                                                                                              });
    if (!loaded_dataset) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, loaded_dataset.error());
        return 1;
    }

    try {
        const dataset::scalar_real::Dataset& dataset = *loaded_dataset;
        hyfluid::train::HyFluid hyfluid{dataset};
        std::uint64_t total_pixel_bytes = 0u;
        for (const hyfluid::train::HyFluid::HostFrameSet& frame_set : hyfluid.host.frame_sets) total_pixel_bytes += frame_set.pixel_bytes;

        std::println("HyFluid ScalarReal dataset loaded.");
        std::println("path: {}", dataset_path.string());
        std::println("scene_scale: {:.6f}", hyfluid.host.scene_scale);
        std::println("near/far: {:.6f} / {:.6f}", hyfluid.host.near, hyfluid.host.far);
        std::println("frame sets: {}", hyfluid.host.frame_sets.size());
        std::println("videos: {}", hyfluid.host.videos.size());
        std::println("frames: {}", hyfluid.host.frames.size());
        std::println("pixel storage: {:.3f} MiB", static_cast<double>(total_pixel_bytes) / 1048576.0);
        std::println("train dataset upload: ok");

        for (const hyfluid::train::HyFluid::HostFrameSet& frame_set : hyfluid.host.frame_sets) {
            std::println("frame_set '{}': {} frames, {} views x {} times, {:.3f} MiB pixels", frame_set.name, frame_set.frame_count, frame_set.view_count, frame_set.time_count, static_cast<double>(frame_set.pixel_bytes) / 1048576.0);
        }

        if (optimize_iterations > 0) {
            const std::string& optimize_frame_set = hyfluid.host.frame_sets.front().name;
            const std::expected<hyfluid::train::OptimizationStats, std::string> stats = hyfluid.optimize(hyfluid::train::OptimizationRequest{
                .frame_set = optimize_frame_set,
                .iterations = optimize_iterations,
            });
            if (!stats) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, stats.error());
                return 1;
            }
            std::println("sampler optimize: frame_set={} step={} rays={}/{} samples={} sample_eff={:.2f}% occupancy_cells={} occupancy_grid={:.2f}% elapsed={:.3f}ms", optimize_frame_set, stats->step, stats->ray_count, stats->rays_per_batch, stats->sample_count, stats->sample_efficiency_ratio * 100.0f, stats->occupancy_grid_occupied_cells, stats->occupancy_grid_ratio * 100.0f, stats->elapsed_ms);
        }

        return 0;
    } catch (const std::exception& error) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, error.what());
        return 1;
    }
}
