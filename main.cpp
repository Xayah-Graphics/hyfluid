import std;
import hyfluid.dataset;
import hyfluid.train;

namespace {
    constexpr std::string_view ansi_reset  = "\x1b[0m";
    constexpr std::string_view ansi_dim    = "\x1b[2m";
    constexpr std::string_view ansi_bold   = "\x1b[1m";
    constexpr std::string_view ansi_cyan   = "\x1b[36m";
    constexpr std::string_view ansi_green  = "\x1b[32m";
    constexpr std::string_view ansi_yellow = "\x1b[33m";
    constexpr std::string_view ansi_red    = "\x1b[31m";

    struct CliOptions final {
        std::filesystem::path dataset_path    = "data/ScalarReal";
        std::filesystem::path test_output_dir = "test";
        std::int32_t steps                    = 50000;
        std::int32_t chunk_steps              = 100;
        std::uint32_t rays_per_step           = 1024u;
        std::uint32_t samples_per_ray         = 192u;
        std::uint32_t test_frame_limit        = 1u;
        float learning_rate                   = 5e-4f;
        std::optional<std::filesystem::path> load_weights_path;
        std::optional<std::filesystem::path> export_weights_path;
    };
} // namespace

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    const std::string executable_name = !arguments.empty() && arguments.front() != nullptr ? std::filesystem::path{arguments.front()}.filename().string() : "hyfluid-app";
    const std::string usage           = std::format(
        R"({}Usage:{}
  {}{}{} {}[options]{}

{}Options:{}
  {}--dataset <path>{}             ScalarReal dataset root
                               {}default:{} data/ScalarReal
  {}--steps <count>{}              total training steps
                               {}default:{} 50000
  {}--chunk-steps <count>{}        training steps per log line
                               {}default:{} 100
  {}--rays-per-step <count>{}      random rays per optimizer step
                               {}default:{} 1024
  {}--samples-per-ray <count>{}    uniform density samples per ray
                               {}default:{} 192
  {}--test-frames <count>{}        test frames per test view
                               {}default:{} 1
  {}--test-output <path>{}         output directory for comparison PNGs
                               {}default:{} test
  {}--learning-rate <value>{}      base RAdam learning rate
                               {}default:{} 5e-4
  {}--load-weights <path>{}        load hyfluid-density.v1 weights before training
  {}--export-weights <path>{}      export hyfluid-density.v1 weights after testing
  {}-h, --help{}                   print this help

{}Examples:{}
  {}{}{} --dataset data/ScalarReal --steps 1000
  {}{}{} --steps 1 --chunk-steps 1 --test-frames 1
)",
        ansi_bold, ansi_reset, ansi_cyan, executable_name, ansi_reset, ansi_dim, ansi_reset, ansi_bold, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_green, ansi_reset, ansi_green, ansi_reset, ansi_bold, ansi_reset, ansi_cyan, executable_name, ansi_reset, ansi_cyan, executable_name, ansi_reset);

    CliOptions cli = {};
    std::optional<std::string> error;

    for (std::size_t i = 1uz; i < arguments.size() && !error.has_value(); ++i) {
        const std::string_view argument{arguments[i]};
        const std::size_t assignment  = argument.find('=');
        const std::string_view option = assignment == std::string_view::npos ? argument : argument.substr(0uz, assignment);
        std::optional<std::string_view> inline_value;
        if (assignment != std::string_view::npos) inline_value = argument.substr(assignment + 1uz);

        if (option == "-h" || option == "--help") {
            if (inline_value.has_value())
                error = std::format("{} does not accept a value.", option);
            else {
                std::println("{}", usage);
                return 0;
            }
        } else if (option == "--dataset" || option == "--test-output" || option == "--load-weights" || option == "--export-weights") {
            std::string_view value;
            if (inline_value.has_value())
                value = *inline_value;
            else if (i + 1uz < arguments.size())
                value = arguments[++i];
            else {
                error = std::format("{} requires a value.", option);
                continue;
            }
            if (value.empty()) {
                error = std::format("{} requires a non-empty value.", option);
                continue;
            }
            if (option == "--dataset") cli.dataset_path = std::filesystem::path{value};
            if (option == "--test-output") cli.test_output_dir = std::filesystem::path{value};
            if (option == "--load-weights") cli.load_weights_path = std::filesystem::path{value};
            if (option == "--export-weights") cli.export_weights_path = std::filesystem::path{value};
        } else if (option == "--steps" || option == "--chunk-steps" || option == "--rays-per-step" || option == "--samples-per-ray" || option == "--test-frames") {
            std::string_view value;
            if (inline_value.has_value())
                value = *inline_value;
            else if (i + 1uz < arguments.size())
                value = arguments[++i];
            else {
                error = std::format("{} requires a value.", option);
                continue;
            }
            std::uint32_t parsed = 0u;
            const auto result    = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0u) {
                error = std::format("{} must be a positive integer.", option);
                continue;
            }
            if (option == "--steps") cli.steps = static_cast<std::int32_t>(parsed);
            if (option == "--chunk-steps") cli.chunk_steps = static_cast<std::int32_t>(parsed);
            if (option == "--rays-per-step") cli.rays_per_step = parsed;
            if (option == "--samples-per-ray") cli.samples_per_ray = parsed;
            if (option == "--test-frames") cli.test_frame_limit = parsed;
        } else if (option == "--learning-rate") {
            std::string_view value;
            if (inline_value.has_value())
                value = *inline_value;
            else if (i + 1uz < arguments.size())
                value = arguments[++i];
            else {
                error = std::format("{} requires a value.", option);
                continue;
            }
            float parsed      = 0.0f;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(parsed) || parsed <= 0.0f)
                error = "--learning-rate must be a positive finite value.";
            else
                cli.learning_rate = parsed;
        } else {
            error = std::format("unknown argument '{}'.", argument);
        }
    }

    if (error.has_value()) {
        std::println("{}error:{} {}\n{}", ansi_red, ansi_reset, *error, usage);
        return 2;
    }
    if (!std::filesystem::is_directory(cli.dataset_path)) {
        std::println("{}error:{} dataset path '{}' is not a directory.", ansi_red, ansi_reset, cli.dataset_path.string());
        return 2;
    }
    if (cli.load_weights_path.has_value() && !std::filesystem::is_regular_file(*cli.load_weights_path)) {
        std::println("{}error:{} weights file '{}' does not exist.", ansi_red, ansi_reset, cli.load_weights_path->string());
        return 2;
    }
    if (cli.export_weights_path.has_value() && !cli.export_weights_path->parent_path().empty() && !std::filesystem::is_directory(cli.export_weights_path->parent_path())) {
        std::println("{}error:{} weights export parent directory '{}' does not exist.", ansi_red, ansi_reset, cli.export_weights_path->parent_path().string());
        return 2;
    }

    const auto config_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} dataset={} steps={} chunk={} rays={} samples={} test_frames={} test_output={} lr={} load_weights={} export_weights={}", ansi_dim, config_timestamp, ansi_reset, ansi_cyan, "CONFIG", ansi_reset, cli.dataset_path.string(), cli.steps, cli.chunk_steps, cli.rays_per_step, cli.samples_per_ray, cli.test_frame_limit, cli.test_output_dir.string(), cli.learning_rate, cli.load_weights_path.has_value() ? cli.load_weights_path->string() : "none", cli.export_weights_path.has_value() ? cli.export_weights_path->string() : "none");

    std::optional<std::string> pipeline_error;
    std::unique_ptr<hyfluid::train::HyFluidDensity> density;
    const auto load_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} loading ScalarReal videos with FFmpeg", ansi_dim, load_timestamp, ansi_reset, ansi_cyan, "INFO", ansi_reset);

    const auto dataset = hyfluid::dataset::load_scalar_real(cli.dataset_path);
    if (!dataset) {
        pipeline_error = dataset.error();
    } else {
        try {
            hyfluid::train::TrainOptions options = {};
            options.rays_per_step                = cli.rays_per_step;
            options.samples_per_ray              = cli.samples_per_ray;
            options.test_frame_limit             = cli.test_frame_limit;
            options.test_output_dir              = cli.test_output_dir;
            options.learning_rate                = cli.learning_rate;
            density                              = std::make_unique<hyfluid::train::HyFluidDensity>(*dataset, options);
        } catch (const std::exception& exception) {
            pipeline_error = exception.what();
        }
    }

    if (!pipeline_error && cli.load_weights_path.has_value()) {
        const auto loaded = density->load_weights(*cli.load_weights_path);
        if (!loaded)
            pipeline_error = loaded.error();
        else
            std::println("{}WEIGHT{} loaded={}", ansi_yellow, ansi_reset, cli.load_weights_path->string());
    }

    float first_loss         = 0.0f;
    float last_loss          = 0.0f;
    float total_train_ms     = 0.0f;
    std::uint32_t final_step = 0u;
    if (!pipeline_error) {
        for (std::int32_t trained_steps = 0; trained_steps < cli.steps;) {
            const std::int32_t requested_steps = std::min(cli.chunk_steps, cli.steps - trained_steps);
            const auto stats                   = density->train(requested_steps);
            if (!stats) {
                pipeline_error = stats.error();
                break;
            }
            if (trained_steps == 0) first_loss = stats->loss;
            last_loss = stats->loss;
            total_train_ms += stats->elapsed_ms;
            final_step = stats->step;
            trained_steps += requested_steps;
            const auto train_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<7}{} step={:>6}/{} loss={:.8f} chunk={:.3f}ms rate={:.2f} step/s rays={} samples={} occ_bin={} occ_cells={} occ={:.6f} occ_update={:.3f}ms skipped={}", ansi_dim, train_timestamp, ansi_reset, ansi_green, "TRAIN", ansi_reset, stats->step, cli.steps, stats->loss, stats->elapsed_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_step, stats->samples_per_ray, stats->occupancy_bin, stats->occupancy_occupied_cells, stats->occupancy_ratio, stats->occupancy_update_ms, stats->occupancy_skipped_samples);
        }
    }

    if (!pipeline_error) {
        const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        std::println("{}[{:%F %T}]{} {}{:<7}{} steps={} first_loss={:.8f} last_loss={:.8f} train={:.3f}s avg={:.2f} step/s", ansi_dim, summary_timestamp, ansi_reset, ansi_bold, "SUMMARY", ansi_reset, final_step, first_loss, last_loss, total_train_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / total_train_ms);
        const auto test = density->test();
        if (!test) {
            pipeline_error = test.error();
        } else {
            const auto test_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<7}{} step={} MSE={:.8f} PSNR={:.2f} images={} saved={} pixels={} output={} test={:.3f}ms", ansi_dim, test_timestamp, ansi_reset, ansi_yellow, "TEST", ansi_reset, test->step, test->mse, test->psnr, test->image_count, test->comparison_image_count, test->pixel_count, test->output_dir.string(), test->elapsed_ms);
        }
    }

    if (!pipeline_error && cli.export_weights_path.has_value()) {
        const auto exported = density->export_weights(*cli.export_weights_path);
        if (!exported)
            pipeline_error = exported.error();
        else
            std::println("{}WEIGHT{} exported={}", ansi_yellow, ansi_reset, cli.export_weights_path->string());
    }

    const auto finish_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    if (!pipeline_error) {
        std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=succeeded", ansi_dim, finish_timestamp, ansi_reset, ansi_bold, "DONE", ansi_reset);
        return 0;
    }
    std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=failed error=\"{}\"", ansi_dim, finish_timestamp, ansi_reset, ansi_red, "ERROR", ansi_reset, *pipeline_error);
    return 1;
}
