import std;
import xayah.util.xcli;
import dataset.scalar_real;
import hyfluid.train;
import hyfluid.inspector;

namespace {
    constexpr std::string_view ansi_reset             = "\x1b[0m";
    constexpr std::string_view ansi_dim               = "\x1b[2m";
    constexpr std::string_view ansi_bold              = "\x1b[1m";
    constexpr std::string_view ansi_cyan              = "\x1b[36m";
    constexpr std::string_view ansi_green             = "\x1b[32m";
    constexpr std::string_view ansi_yellow            = "\x1b[33m";
    constexpr std::string_view ansi_red               = "\x1b[31m";
    constexpr std::string_view ansi_evaluation_badge  = "\x1b[1;37;45m";
    constexpr std::string_view ansi_evaluation_metric = "\x1b[1;95m";
    constexpr float scene_scale                       = 1.0f;

    [[nodiscard]] bool is_scalar_real_frame_set(const std::string_view name) {
        return name == "train" || name == "test";
    }

    void append_unique_frame_set(std::vector<std::string>& frame_sets, const std::string_view name) {
        for (const std::string& frame_set : frame_sets)
            if (frame_set == name) return;
        frame_sets.emplace_back(name);
    }

    [[nodiscard]] std::string join_frame_sets(const std::vector<std::string>& frame_sets) {
        std::string result;
        for (const std::string& frame_set : frame_sets) {
            if (!result.empty()) result += ",";
            result += frame_set;
        }
        return result;
    }

    [[nodiscard]] std::filesystem::path automatic_evaluation_output_dir(const std::string_view frame_set, const std::uint32_t step) {
        return std::filesystem::path{"logs"} / "hyfluid-density-eval" / std::format("{}-step-{}", frame_set, step);
    }

    void write_json_string(std::ostream& output, const std::string_view text) {
        output << '"';
        for (const char character : text) {
            switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20u)
                    output << std::format("\\u{:04x}", static_cast<unsigned int>(static_cast<unsigned char>(character)));
                else
                    output << character;
                break;
            }
        }
        output << '"';
    }

    void write_training_log_entry(std::ostream& output, const std::string_view frame_set, const std::int32_t chunk_iterations, const hyfluid::train::OptimizationStats& stats, const hyfluid::inspector::TrainingBatchDiagnostics& batch, const hyfluid::inspector::TrainingModelDiagnostics& model) {
        output << std::setprecision(9);
        output << "{\"frame_set\":";
        write_json_string(output, frame_set);
        output << ",\"chunk_iterations\":" << chunk_iterations;
        output << ",\"step\":" << stats.step;
        output << ",\"loss\":" << stats.loss;
        output << ",\"psnr\":" << stats.psnr;
        output << ",\"ray_count\":" << stats.ray_count;
        output << ",\"rays_per_batch\":" << stats.rays_per_batch;
        output << ",\"next_rays_per_batch\":" << stats.next_rays_per_batch;
        output << ",\"sample_count\":" << stats.sample_count;
        output << ",\"sample_count_before_compaction\":" << stats.sample_count_before_compaction;
        output << ",\"sample_efficiency_ratio\":" << stats.sample_efficiency_ratio;
        output << ",\"coord_min\":[" << batch.sample_coord_min[0u] << "," << batch.sample_coord_min[1u] << "," << batch.sample_coord_min[2u] << "]";
        output << ",\"coord_max\":[" << batch.sample_coord_max[0u] << "," << batch.sample_coord_max[1u] << "," << batch.sample_coord_max[2u] << "]";
        output << ",\"time_min\":" << batch.time_min;
        output << ",\"time_max\":" << batch.time_max;
        output << ",\"dt_metric_min\":" << batch.dt_metric_min;
        output << ",\"dt_metric_mean\":" << batch.dt_metric_mean;
        output << ",\"dt_metric_max\":" << batch.dt_metric_max;
        output << ",\"metric_per_field_unit_min\":" << batch.metric_per_field_unit_min;
        output << ",\"metric_per_field_unit_mean\":" << batch.metric_per_field_unit_mean;
        output << ",\"metric_per_field_unit_max\":" << batch.metric_per_field_unit_max;
        output << ",\"global_rgb_param\":" << model.global_rgb_param;
        output << ",\"global_rgb_color\":" << model.global_rgb_color;
        output << ",\"global_rgb_gradient\":" << model.global_rgb_gradient;
        output << ",\"occupancy_grid_occupied_cells\":" << stats.occupancy_grid_occupied_cells;
        output << ",\"occupancy_grid_ratio\":" << stats.occupancy_grid_ratio;
        output << ",\"occupancy_grid_updated_bin\":" << stats.occupancy_grid_updated_bin;
        output << ",\"occupancy_grid_updated_bin_occupied_cells\":" << stats.occupancy_grid_updated_bin_occupied_cells;
        output << ",\"elapsed_ms\":" << stats.elapsed_ms;
        output << "}\n";
    }
} // namespace

int main(const int argc, const char* const* const argv) {
    const std::span arguments{argv, static_cast<std::size_t>(argc)};

    std::filesystem::path dataset_path;
    std::string optimize_frame_set = "train";
    std::optional<std::string> evaluate_frame_set;
    std::int32_t steps     = 0;
    std::int32_t log_every = 100;
    std::optional<std::filesystem::path> training_log_path;
    std::optional<std::filesystem::path> load_weights_path;
    std::optional<std::filesystem::path> save_weights_path;

    xayah::util::Command command = xayah::util::Command{"ScalarReal dynamic dataset runner for HyFluid."}
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
                                         .long_name    = "optimize",
                                         .value_name   = "frame-set",
                                         .description  = "Frame set used for density-stage optimization.",
                                         .default_text = "train",
                                     },
                                     optimize_frame_set)
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "steps",
                                         .value_name   = "count",
                                         .description  = "Total density-stage optimization steps.",
                                         .default_text = "0",
                                     },
                                     steps, xayah::util::NumericRule{.minimum = 0.0})
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "log-every",
                                         .value_name   = "count",
                                         .description  = "Optimization steps per terminal and JSONL log entry.",
                                         .default_text = "100",
                                     },
                                     log_every, xayah::util::NumericRule{.minimum = 1.0})
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "training-log",
                                         .value_name   = "path",
                                         .description  = "Write density-stage optimization metrics as JSONL.",
                                         .default_text = "none",
                                     },
                                     training_log_path)
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "evaluate",
                                         .value_name   = "frame-set",
                                         .description  = "Frame set to render and evaluate after optional optimization.",
                                         .default_text = "off",
                                     },
                                     evaluate_frame_set)
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "load-weights",
                                         .value_name   = "path",
                                         .description  = "Load HyFluid density safetensors weights before optimization or evaluation.",
                                         .default_text = "none",
                                     },
                                     load_weights_path, xayah::util::PathRule{.requirement = xayah::util::PathRequirement::existing_file})
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "save-weights",
                                         .value_name   = "path",
                                         .description  = "Save final HyFluid density safetensors weights.",
                                         .default_text = "none",
                                     },
                                     save_weights_path, xayah::util::PathRule{.requirement = xayah::util::PathRequirement::existing_parent_directory})
                                 | xayah::util::example("data/ScalarReal")
                                 | xayah::util::example("data/ScalarReal --steps 10000")
                                 | xayah::util::example("data/ScalarReal --optimize test --steps 10000")
                                 | xayah::util::example("data/ScalarReal --steps 10000 --evaluate test")
                                 | xayah::util::example("data/ScalarReal --load-weights logs/hyfluid-density.safetensors --evaluate test");

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

    const auto path_validation = command.validate();
    if (!path_validation) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, path_validation.error());
        return 2;
    }

    const bool optimization_enabled = steps > 0;
    const bool evaluation_enabled   = evaluate_frame_set.has_value();
    const bool training_log_enabled = command.option_provided("training-log");

    std::optional<std::string> cli_error;
    if (!is_scalar_real_frame_set(optimize_frame_set))
        cli_error = std::format("--optimize must be one of train or test; got '{}'.", optimize_frame_set);
    else if (evaluation_enabled && !is_scalar_real_frame_set(*evaluate_frame_set))
        cli_error = std::format("--evaluate must be one of train or test; got '{}'.", *evaluate_frame_set);
    else if (command.option_provided("optimize") && !optimization_enabled)
        cli_error = "--optimize requires --steps with a positive count.";
    else if (training_log_enabled && !optimization_enabled)
        cli_error = "--training-log requires --steps with a positive count.";
    else if (evaluation_enabled && !optimization_enabled && !load_weights_path.has_value())
        cli_error = "--evaluate without optimization requires --load-weights.";
    else if (save_weights_path.has_value() && !optimization_enabled && !load_weights_path.has_value())
        cli_error = "--save-weights requires optimization or --load-weights.";
    if (cli_error.has_value()) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, *cli_error);
        return 2;
    }

    std::vector<std::string> requested_frame_sets;
    if (optimization_enabled) append_unique_frame_set(requested_frame_sets, optimize_frame_set);
    if (evaluation_enabled) append_unique_frame_set(requested_frame_sets, *evaluate_frame_set);
    if (requested_frame_sets.empty()) requested_frame_sets.push_back("train");

    const std::string frame_set_stage    = join_frame_sets(requested_frame_sets);
    const std::string optimize_stage     = optimization_enabled ? optimize_frame_set : std::string{"off"};
    const std::string evaluation_stage   = evaluation_enabled ? std::format("{},output=auto", *evaluate_frame_set) : std::string{"off"};
    const std::string training_log_stage = training_log_enabled ? training_log_path->string() : std::string{"off"};
    const auto config_timestamp          = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<8}{} dataset={} scene_scale={} frame_sets={} optimize={} steps={} log_every={} evaluation={} training_log={} load_weights={} save_weights={}", ansi_dim, config_timestamp, ansi_reset, ansi_cyan, "CONFIG", ansi_reset, dataset_path.string(), scene_scale, frame_set_stage, optimize_stage, steps, log_every, evaluation_stage, training_log_stage, load_weights_path.has_value() ? load_weights_path->string() : "none", save_weights_path.has_value() ? save_weights_path->string() : "none");

    const auto load_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<8}{} loading dataset", ansi_dim, load_timestamp, ansi_reset, ansi_cyan, "INFO", ansi_reset);

    const std::expected<dataset::scalar_real::Dataset, std::string> loaded_dataset = dataset::scalar_real::load(dataset_path, dataset::scalar_real::LoadRequest{
                                                                                                                                  .frame_sets  = requested_frame_sets,
                                                                                                                                  .scene_scale = scene_scale,
                                                                                                                              });
    if (!loaded_dataset) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, loaded_dataset.error());
        return 1;
    }

    try {
        const dataset::scalar_real::Dataset& dataset = *loaded_dataset;
        hyfluid::train::HyFluid hyfluid{dataset};

        const auto initialized_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        std::println("{}[{:%F %T}]{} {}{:<8}{} initialized frame_sets={}", ansi_dim, initialized_timestamp, ansi_reset, ansi_cyan, "INFO", ansi_reset, frame_set_stage);

        if (load_weights_path.has_value()) {
            const std::expected<void, std::string> loaded_weights = hyfluid.load_weights(*load_weights_path);
            if (!loaded_weights) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, loaded_weights.error());
                return 1;
            }
            const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<8}{} loaded={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, load_weights_path->string());
        }

        std::optional<float> first_loss;
        float last_loss          = 0.0f;
        float optimize_ms        = 0.0f;
        std::uint32_t final_step = hyfluid.host.current_step;

        if (optimization_enabled) {
            std::ofstream training_log;
            std::optional<hyfluid::inspector::Inspector> inspector;
            if (training_log_enabled) {
                if (training_log_path->has_parent_path()) std::filesystem::create_directories(training_log_path->parent_path());
                training_log.open(*training_log_path, std::ios::binary | std::ios::trunc);
                if (!training_log) {
                    std::println("{}error:{} failed to open training log '{}'.", ansi_red, ansi_reset, training_log_path->string());
                    return 1;
                }
                inspector.emplace(hyfluid);
            }

            for (std::int32_t optimized_steps = 0; optimized_steps < steps;) {
                const std::int32_t chunk_iterations                                       = std::min(log_every, steps - optimized_steps);
                const std::expected<hyfluid::train::OptimizationStats, std::string> stats = hyfluid.optimize(hyfluid::train::OptimizationRequest{
                    .frame_set  = optimize_frame_set,
                    .iterations = chunk_iterations,
                });
                if (!stats) {
                    std::println("{}error:{} {}", ansi_red, ansi_reset, stats.error());
                    return 1;
                }

                if (!first_loss.has_value()) first_loss = stats->loss;
                last_loss = stats->loss;
                optimize_ms += stats->elapsed_ms;
                final_step = stats->step;
                optimized_steps += chunk_iterations;

                const auto optimize_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<8}{} frame_set={} step={:>6}/{} loss={:>10.6f} psnr={:>5.2f} chunk={:>8.3f}ms rate={:>7.2f} step/s next_rays={:>6} samples={:>7}/{:<7} sample_eff={:>6.2f}% occupancy_grid={:>6.2f}% updated_bin={}", ansi_dim, optimize_timestamp, ansi_reset, ansi_green, "OPTIMIZE", ansi_reset, optimize_frame_set, final_step, steps, stats->loss, stats->psnr, stats->elapsed_ms, static_cast<float>(chunk_iterations) * 1000.0f / stats->elapsed_ms, stats->next_rays_per_batch, stats->sample_count, stats->sample_count_before_compaction, stats->sample_efficiency_ratio * 100.0f, stats->occupancy_grid_ratio * 100.0f, stats->occupancy_grid_updated_bin);

                if (training_log_enabled) {
                    const hyfluid::inspector::TrainingBatchDiagnostics batch_diagnostics = inspector->training_batch_diagnostics();
                    const hyfluid::inspector::TrainingModelDiagnostics model_diagnostics = inspector->training_model_diagnostics();
                    write_training_log_entry(training_log, optimize_frame_set, chunk_iterations, *stats, batch_diagnostics, model_diagnostics);
                    training_log.flush();
                }
            }
        }

        std::optional<hyfluid::train::EvaluationStats> final_evaluation;
        if (evaluation_enabled) {
            const std::filesystem::path evaluation_output_dir = automatic_evaluation_output_dir(*evaluate_frame_set, hyfluid.host.current_step);
            const std::expected<hyfluid::train::EvaluationStats, std::string> evaluation = hyfluid.evaluate(hyfluid::train::EvaluationRequest{
                .frame_set  = *evaluate_frame_set,
                .output_dir = evaluation_output_dir,
            });
            if (!evaluation) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, evaluation.error());
                return 1;
            }
            final_evaluation                = *evaluation;
            const auto evaluation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3}/{} pixels={} output={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, evaluation->rendered_image_count, evaluation->image_count, evaluation->pixel_count, evaluation->output_dir.string(), evaluation->elapsed_ms);
        }

        if (optimization_enabled || final_evaluation.has_value()) {
            const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            if (optimization_enabled && final_evaluation.has_value()) {
                std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s evaluation={}:{:.8f}@{} psnr={:.2f}", ansi_dim, summary_timestamp, ansi_reset, ansi_bold, "SUMMARY", ansi_reset, final_step, *first_loss, last_loss, optimize_ms * 0.001f, static_cast<float>(steps) * 1000.0f / optimize_ms, final_evaluation->frame_set, final_evaluation->mse, final_evaluation->step, final_evaluation->psnr);
            } else if (optimization_enabled) {
                std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s evaluation=off", ansi_dim, summary_timestamp, ansi_reset, ansi_bold, "SUMMARY", ansi_reset, final_step, *first_loss, last_loss, optimize_ms * 0.001f, static_cast<float>(steps) * 1000.0f / optimize_ms);
            } else {
                std::println("{}[{:%F %T}]{} {}{:<8}{} optimize=off evaluation={}:{:.8f}@{} psnr={:.2f}", ansi_dim, summary_timestamp, ansi_reset, ansi_bold, "SUMMARY", ansi_reset, final_evaluation->frame_set, final_evaluation->mse, final_evaluation->step, final_evaluation->psnr);
            }
        }

        if (save_weights_path.has_value()) {
            const std::expected<void, std::string> saved_weights = hyfluid.export_weights(*save_weights_path);
            if (!saved_weights) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, saved_weights.error());
                return 1;
            }
            const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<8}{} saved={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, save_weights_path->string());
        }

        return 0;
    } catch (const std::exception& error) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, error.what());
        return 1;
    }
}
