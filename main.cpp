import std;
import xayah.util.xcli;
import dataset.scalar_real;
import hyfluid.train;
import hyfluid.inspector;

int main(const int argc, const char* const* const argv) {
    const std::span arguments{argv, static_cast<std::size_t>(argc)};
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

    std::filesystem::path dataset_path;
    std::vector<std::string> requested_frame_sets;
    std::int32_t optimize_iterations = 0;
    std::int32_t log_every = 100;
    std::filesystem::path training_log_path;
    std::filesystem::path evaluation_output_path;
    std::optional<std::filesystem::path> load_weights_path;
    std::optional<std::filesystem::path> save_weights_path;
    bool include_test_in_training = false;

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
                                         .description  = "Run density-stage optimization smoke iterations on the first loaded frame set.",
                                         .default_text = "0",
                                     },
                                     optimize_iterations)
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "log-every",
                                         .value_name   = "iterations",
                                         .description  = "Print and optionally write one training log entry every N optimization iterations.",
                                         .default_text = "100",
                                     },
                                     log_every)
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
                                         .long_name    = "evaluate-output",
                                         .value_name   = "directory",
                                         .description  = "After density-stage optimization, render all test frames as RGB PNG files and write metrics.json.",
                                         .default_text = "none",
                                     },
                                     evaluation_output_path)
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
                                         .description  = "Save final HyFluid density safetensors weights after optimization and evaluation.",
                                         .default_text = "none",
                                     },
                                     save_weights_path, xayah::util::PathRule{.requirement = xayah::util::PathRequirement::existing_parent_directory})
                                 | xayah::util::option(
                                     xayah::util::OptionSpec{
                                         .long_name    = "include-test-in-training",
                                         .description  = "Include the ScalarReal test frame set in density optimization before test evaluation.",
                                         .show_default = false,
                                     },
                                     include_test_in_training)
                                 | xayah::util::example("data/ScalarReal")
                                 | xayah::util::example("data/ScalarReal --frame-set train --frame-set test")
                                 | xayah::util::example("data/ScalarReal --frame-set train --optimize 1")
                                 | xayah::util::example("data/ScalarReal --frame-set train --optimize 1 --save-weights logs/hyfluid-density.safetensors")
                                 | xayah::util::example("data/ScalarReal --frame-set train --load-weights logs/hyfluid-density.safetensors --optimize 1")
                                 | xayah::util::example("data/ScalarReal --optimize 10000 --log-every 100 --training-log logs/hyfluid-density.jsonl --evaluate-output logs/hyfluid-density-test")
                                 | xayah::util::example("data/ScalarReal --optimize 10000 --include-test-in-training --training-log logs/hyfluid-density-train-plus-test.jsonl --evaluate-output logs/hyfluid-density-train-plus-test");

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

    const bool writes_evaluation_output = command.option_provided("evaluate-output");
    if (!command.option_provided("frame-set")) requested_frame_sets.push_back("train");
    if (writes_evaluation_output) {
        bool has_train = false;
        bool has_test = false;
        for (const std::string& frame_set_name : requested_frame_sets) {
            if (frame_set_name == "train") has_train = true;
            if (frame_set_name == "test") has_test = true;
        }
        if (!has_train) requested_frame_sets.push_back("train");
        if (!has_test) requested_frame_sets.push_back("test");
    }
    if (command.option_provided("optimize") && optimize_iterations < 1) {
        std::println("{}error:{} --optimize must be positive when provided.", ansi_red, ansi_reset);
        return 2;
    }
    if (writes_evaluation_output && optimize_iterations < 1) {
        std::println("{}error:{} --evaluate-output requires --optimize with a positive iteration count.", ansi_red, ansi_reset);
        return 2;
    }
    if (writes_evaluation_output && evaluation_output_path.empty()) {
        std::println("{}error:{} --evaluate-output directory must not be empty when provided.", ansi_red, ansi_reset);
        return 2;
    }
    if (include_test_in_training && !writes_evaluation_output) {
        std::println("{}error:{} --include-test-in-training requires --evaluate-output.", ansi_red, ansi_reset);
        return 2;
    }
    if (log_every < 1) {
        std::println("{}error:{} --log-every must be positive.", ansi_red, ansi_reset);
        return 2;
    }
    if (command.option_provided("training-log") && training_log_path.empty()) {
        std::println("{}error:{} --training-log path must not be empty when provided.", ansi_red, ansi_reset);
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

    std::string frame_set_stage;
    for (std::size_t i = 0uz; i < requested_frame_sets.size(); ++i) {
        if (i != 0uz) frame_set_stage += "+";
        frame_set_stage += requested_frame_sets[i];
    }
    std::string optimize_stage = "off";
    if (optimize_iterations > 0) {
        if (writes_evaluation_output) {
            optimize_stage = "train";
            if (include_test_in_training) optimize_stage += "+test";
        } else {
            optimize_stage = requested_frame_sets.front();
        }
    }
    const std::string evaluation_stage = writes_evaluation_output ? std::format("test,output={}", evaluation_output_path.string()) : std::string{"off"};
    const std::string training_log_stage = command.option_provided("training-log") ? training_log_path.string() : std::string{"off"};

    const auto config_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<8}{} dataset={} scene_scale={} frame_sets={} optimize={} steps={} log_every={} evaluation={} training_log={} load_weights={} save_weights={}",
                 ansi_dim,
                 config_timestamp,
                 ansi_reset,
                 ansi_cyan,
                 "CONFIG",
                 ansi_reset,
                 dataset_path.string(),
                 scene_scale,
                 frame_set_stage,
                 optimize_stage,
                 optimize_iterations,
                 log_every,
                 evaluation_stage,
                 training_log_stage,
                 load_weights_path.has_value() ? load_weights_path->string() : "none",
                 save_weights_path.has_value() ? save_weights_path->string() : "none");

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
        hyfluid::inspector::Inspector inspector{hyfluid};
        if (load_weights_path.has_value()) {
            const std::expected<void, std::string> loaded_weights = hyfluid.load_weights(*load_weights_path);
            if (!loaded_weights) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, loaded_weights.error());
                return 1;
            }
            const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<8}{} loaded={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, load_weights_path->string());
        }

        if (optimize_iterations > 0) {
            std::vector<std::string> optimize_frame_sets;
            if (writes_evaluation_output) {
                optimize_frame_sets.push_back("train");
                if (include_test_in_training) optimize_frame_sets.push_back("test");
            } else {
                optimize_frame_sets.push_back(hyfluid.host.frame_sets.front().name);
            }
            std::string optimize_frame_set_label;
            for (std::size_t i = 0uz; i < optimize_frame_sets.size(); ++i) {
                if (i != 0uz) optimize_frame_set_label += "+";
                optimize_frame_set_label += optimize_frame_sets[i];
            }
            std::vector<std::uint64_t> optimize_frame_set_weights;
            std::uint64_t optimize_frame_set_weight_sum = 0u;
            optimize_frame_set_weights.reserve(optimize_frame_sets.size());
            for (const std::string& optimize_frame_set : optimize_frame_sets) {
                bool found_frame_set = false;
                for (const hyfluid::train::HyFluid::HostFrameSet& frame_set : hyfluid.host.frame_sets) {
                    if (frame_set.name != optimize_frame_set) continue;
                    const std::uint64_t frame_count = static_cast<std::uint64_t>(frame_set.view_count) * frame_set.time_count;
                    if (frame_count == 0u) {
                        std::println("{}error:{} requested optimization frame set '{}' has no frames.", ansi_red, ansi_reset, optimize_frame_set);
                        return 1;
                    }
                    optimize_frame_set_weights.push_back(frame_count);
                    optimize_frame_set_weight_sum += frame_count;
                    found_frame_set = true;
                    break;
                }
                if (!found_frame_set) {
                    std::println("{}error:{} requested optimization frame set '{}' is not loaded.", ansi_red, ansi_reset, optimize_frame_set);
                    return 1;
                }
            }
            std::ofstream training_log;
            if (command.option_provided("training-log")) {
                if (training_log_path.has_parent_path()) std::filesystem::create_directories(training_log_path.parent_path());
                training_log.open(training_log_path, std::ios::binary | std::ios::trunc);
                if (!training_log) {
                    std::println("{}error:{} failed to open training log '{}'.", ansi_red, ansi_reset, training_log_path.string());
                    return 1;
                }
            }

            std::vector<std::pair<std::int32_t, float>> recent_psnr;
            if (training_log) recent_psnr.reserve(1000uz);
            std::int32_t recent_psnr_steps = 0;
            std::int32_t remaining_iterations = optimize_iterations;
            bool first_optimization_chunk = true;
            float first_loss = 0.0f;
            float last_loss = 0.0f;
            float optimize_ms = 0.0f;
            std::uint32_t final_step = 0u;
            std::optional<hyfluid::train::EvaluationStats> final_evaluation;
            while (remaining_iterations > 0) {
                const std::int32_t chunk_iterations = std::min(log_every, remaining_iterations);
                std::optional<hyfluid::train::OptimizationStats> chunk_stats;
                std::int32_t distributed_iterations = 0;
                std::uint64_t distributed_weight = 0u;
                float chunk_ms = 0.0f;
                for (std::size_t frame_set_index = 0uz; frame_set_index < optimize_frame_sets.size(); ++frame_set_index) {
                    const std::uint64_t next_distributed_weight = distributed_weight + optimize_frame_set_weights[frame_set_index];
                    const auto previous_iteration_boundary = static_cast<std::int32_t>(static_cast<std::uint64_t>(chunk_iterations) * distributed_weight / optimize_frame_set_weight_sum);
                    const auto next_iteration_boundary = static_cast<std::int32_t>(static_cast<std::uint64_t>(chunk_iterations) * next_distributed_weight / optimize_frame_set_weight_sum);
                    const std::int32_t frame_set_iterations = next_iteration_boundary - previous_iteration_boundary;
                    distributed_weight = next_distributed_weight;
                    if (frame_set_iterations == 0) continue;
                    distributed_iterations += frame_set_iterations;
                    const std::expected<hyfluid::train::OptimizationStats, std::string> stats = hyfluid.optimize(hyfluid::train::OptimizationRequest{
                        .frame_set = optimize_frame_sets[frame_set_index],
                        .iterations = frame_set_iterations,
                    });
                    if (!stats) {
                        std::println("{}error:{} {}", ansi_red, ansi_reset, stats.error());
                        return 1;
                    }
                    chunk_ms += stats->elapsed_ms;
                    chunk_stats = *stats;
                }
                if (!chunk_stats.has_value() || distributed_iterations != chunk_iterations) {
                    std::println("{}error:{} optimization iteration distribution failed.", ansi_red, ansi_reset);
                    return 1;
                }
                const hyfluid::train::OptimizationStats& stats = *chunk_stats;

                if (first_optimization_chunk) first_loss = stats.loss;
                first_optimization_chunk = false;
                last_loss = stats.loss;
                optimize_ms += chunk_ms;
                final_step = stats.step;

                const auto optimize_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<8}{} frame_set={} step={:>6}/{} loss={:>10.6f} psnr={:>5.2f} chunk={:>8.3f}ms rate={:>7.2f} step/s next_rays={:>6} samples={:>7}/{:<7} sample_eff={:>6.2f}% occupancy_grid={:>6.2f}% updated_bin={}",
                             ansi_dim,
                             optimize_timestamp,
                             ansi_reset,
                             ansi_green,
                             "OPTIMIZE",
                             ansi_reset,
                             optimize_frame_set_label,
                             stats.step,
                             optimize_iterations,
                             stats.loss,
                             stats.psnr,
                             chunk_ms,
                             static_cast<float>(chunk_iterations) * 1000.0f / chunk_ms,
                             stats.next_rays_per_batch,
                             stats.sample_count,
                             stats.sample_count_before_compaction,
                             stats.sample_efficiency_ratio * 100.0f,
                             stats.occupancy_grid_ratio * 100.0f,
                             stats.occupancy_grid_updated_bin);

                if (training_log) {
                    recent_psnr.push_back({chunk_iterations, stats.psnr});
                    recent_psnr_steps += chunk_iterations;
                    while (recent_psnr_steps > 1000 && !recent_psnr.empty()) {
                        const std::int32_t excess_steps = recent_psnr_steps - 1000;
                        if (recent_psnr.front().first <= excess_steps) {
                            recent_psnr_steps -= recent_psnr.front().first;
                            recent_psnr.erase(recent_psnr.begin());
                        } else {
                            recent_psnr.front().first -= excess_steps;
                            recent_psnr_steps -= excess_steps;
                        }
                    }
                    double recent_psnr_sum = 0.0;
                    for (const std::pair<std::int32_t, float>& value : recent_psnr) recent_psnr_sum += static_cast<double>(value.first) * static_cast<double>(value.second);
                    const auto recent_psnr_mean = static_cast<float>(recent_psnr_sum / static_cast<double>(recent_psnr_steps));
                    const hyfluid::inspector::TrainingBatchDiagnostics batch_diagnostics = inspector.training_batch_diagnostics();
                    const hyfluid::inspector::TrainingModelDiagnostics model_diagnostics = inspector.training_model_diagnostics();
                    training_log << std::format("{{\"step\":{},\"loss\":{:.9g},\"psnr\":{:.9g},\"recent_psnr\":{:.9g},\"ray_count\":{},\"rays_per_batch\":{},\"next_rays_per_batch\":{},\"sample_count\":{},\"sample_count_before_compaction\":{},\"sample_efficiency_ratio\":{:.9g},\"coord_min\":[{:.9g},{:.9g},{:.9g}],\"coord_max\":[{:.9g},{:.9g},{:.9g}],\"time_min\":{:.9g},\"time_max\":{:.9g},\"dt_metric_min\":{:.9g},\"dt_metric_mean\":{:.9g},\"dt_metric_max\":{:.9g},\"metric_per_field_unit_min\":{:.9g},\"metric_per_field_unit_mean\":{:.9g},\"metric_per_field_unit_max\":{:.9g},\"global_rgb_param\":{:.9g},\"global_rgb_color\":{:.9g},\"global_rgb_gradient\":{:.9g},\"occupancy_grid_occupied_cells\":{},\"occupancy_grid_ratio\":{:.9g},\"occupancy_grid_updated_bin\":{},\"occupancy_grid_updated_bin_occupied_cells\":{},\"elapsed_ms\":{:.9g}}}\n",
                                                stats.step,
                                                stats.loss,
                                                stats.psnr,
                                                recent_psnr_mean,
                                                stats.ray_count,
                                                stats.rays_per_batch,
                                                stats.next_rays_per_batch,
                                                stats.sample_count,
                                                stats.sample_count_before_compaction,
                                                stats.sample_efficiency_ratio,
                                                batch_diagnostics.sample_coord_min[0u],
                                                batch_diagnostics.sample_coord_min[1u],
                                                batch_diagnostics.sample_coord_min[2u],
                                                batch_diagnostics.sample_coord_max[0u],
                                                batch_diagnostics.sample_coord_max[1u],
                                                batch_diagnostics.sample_coord_max[2u],
                                                batch_diagnostics.time_min,
                                                batch_diagnostics.time_max,
                                                batch_diagnostics.dt_metric_min,
                                                batch_diagnostics.dt_metric_mean,
                                                batch_diagnostics.dt_metric_max,
                                                batch_diagnostics.metric_per_field_unit_min,
                                                batch_diagnostics.metric_per_field_unit_mean,
                                                batch_diagnostics.metric_per_field_unit_max,
                                                model_diagnostics.global_rgb_param,
                                                model_diagnostics.global_rgb_color,
                                                model_diagnostics.global_rgb_gradient,
                                                stats.occupancy_grid_occupied_cells,
                                                stats.occupancy_grid_ratio,
                                                stats.occupancy_grid_updated_bin,
                                                stats.occupancy_grid_updated_bin_occupied_cells,
                                                chunk_ms);
                    training_log.flush();
                }
                remaining_iterations -= chunk_iterations;
            }

            if (writes_evaluation_output) {
                const std::expected<hyfluid::train::EvaluationStats, std::string> evaluation = hyfluid.evaluate(hyfluid::train::EvaluationRequest{
                    .frame_set = "test",
                    .output_dir = evaluation_output_path,
                });
                if (!evaluation) {
                    std::println("{}error:{} {}", ansi_red, ansi_reset, evaluation.error());
                    return 1;
                }
                final_evaluation = *evaluation;
                const auto evaluation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3}/{} pixels={} output={} eval={:>8.3f}ms",
                             ansi_dim,
                             evaluation_timestamp,
                             ansi_reset,
                             ansi_evaluation_badge,
                             "EVAL",
                             ansi_reset,
                             evaluation->frame_set,
                             evaluation->step,
                             ansi_evaluation_metric,
                             evaluation->mse,
                             ansi_reset,
                             ansi_cyan,
                             evaluation->psnr,
                             ansi_reset,
                             evaluation->rendered_image_count,
                             evaluation->image_count,
                             evaluation->pixel_count,
                             evaluation->output_dir.string(),
                             evaluation->elapsed_ms);
            }
            const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            if (final_evaluation.has_value()) {
                std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s evaluation={}:{:.8f}@{} psnr={:.2f}",
                             ansi_dim,
                             summary_timestamp,
                             ansi_reset,
                             ansi_bold,
                             "SUMMARY",
                             ansi_reset,
                             final_step,
                             first_loss,
                             last_loss,
                             optimize_ms * 0.001f,
                             static_cast<float>(optimize_iterations) * 1000.0f / optimize_ms,
                             final_evaluation->frame_set,
                             final_evaluation->mse,
                             final_evaluation->step,
                             final_evaluation->psnr);
            } else {
                std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s evaluation=off",
                             ansi_dim,
                             summary_timestamp,
                             ansi_reset,
                             ansi_bold,
                             "SUMMARY",
                             ansi_reset,
                             final_step,
                             first_loss,
                             last_loss,
                             optimize_ms * 0.001f,
                             static_cast<float>(optimize_iterations) * 1000.0f / optimize_ms);
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
