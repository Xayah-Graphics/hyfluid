import std;
import xayah.util.xcli;
import dataset.scalar_real;
import hyfluid.train;
import hyfluid.inspector;

int main(const int argc, const char* const* const argv) {
    const std::span arguments{argv, static_cast<std::size_t>(argc)};
    constexpr std::string_view ansi_reset = "\x1b[0m";
    constexpr std::string_view ansi_red   = "\x1b[31m";

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
        hyfluid::inspector::Inspector inspector{hyfluid};
        if (load_weights_path.has_value()) {
            const std::expected<void, std::string> loaded_weights = hyfluid.load_weights(*load_weights_path);
            if (!loaded_weights) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, loaded_weights.error());
                return 1;
            }
            std::println("weights loaded: {}", load_weights_path->string());
        }
        const hyfluid::inspector::TrainingDomainView training_domain = inspector.training_domain_view();
        std::uint64_t total_pixel_bytes = 0u;
        for (const hyfluid::train::HyFluid::HostFrameSet& frame_set : hyfluid.host.frame_sets) total_pixel_bytes += frame_set.pixel_bytes;

        std::println("HyFluid ScalarReal dataset loaded.");
        std::println("path: {}", dataset_path.string());
        std::println("scene_scale: {:.6f}", dataset.scene_scale);
        std::println("near/far: {:.6f} / {:.6f}", dataset.near, dataset.far);
        std::println("field_metric_extent: [{:.6f}, {:.6f}, {:.6f}]", training_domain.field_metric_extent[0u], training_domain.field_metric_extent[1u], training_domain.field_metric_extent[2u]);
        std::println("frame sets: {}", hyfluid.host.frame_sets.size());
        std::println("videos: {}", dataset.videos.size());
        std::println("frames: {}", hyfluid.host.frames.size());
        std::println("pixel storage: {:.3f} MiB", static_cast<double>(total_pixel_bytes) / 1048576.0);
        std::println("train dataset upload: ok");

        for (const hyfluid::train::HyFluid::HostFrameSet& frame_set : hyfluid.host.frame_sets) {
            std::println("frame_set '{}': {} frames, {} views x {} times, {:.3f} MiB pixels", frame_set.name, frame_set.frame_count, frame_set.view_count, frame_set.time_count, static_cast<double>(frame_set.pixel_bytes) / 1048576.0);
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
                    if (frame_set.frame_count == 0u) {
                        std::println("{}error:{} requested optimization frame set '{}' has no frames.", ansi_red, ansi_reset, optimize_frame_set);
                        return 1;
                    }
                    optimize_frame_set_weights.push_back(static_cast<std::uint64_t>(frame_set.frame_count));
                    optimize_frame_set_weight_sum += static_cast<std::uint64_t>(frame_set.frame_count);
                    found_frame_set = true;
                    break;
                }
                if (!found_frame_set) {
                    std::println("{}error:{} requested optimization frame set '{}' is not loaded.", ansi_red, ansi_reset, optimize_frame_set);
                    return 1;
                }
            }
            std::string optimize_frame_set_weight_label;
            for (std::size_t i = 0uz; i < optimize_frame_sets.size(); ++i) {
                if (i != 0uz) optimize_frame_set_weight_label += ", ";
                optimize_frame_set_weight_label += std::format("{}={}", optimize_frame_sets[i], optimize_frame_set_weights[i]);
            }
            std::println("optimization frame sets: {} ({})", optimize_frame_set_label, optimize_frame_set_weight_label);
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
            recent_psnr.reserve(1000uz);
            std::int32_t recent_psnr_steps = 0;
            std::int32_t remaining_iterations = optimize_iterations;
            while (remaining_iterations > 0) {
                const std::int32_t chunk_iterations = std::min(log_every, remaining_iterations);
                std::optional<hyfluid::train::OptimizationStats> chunk_stats;
                std::int32_t distributed_iterations = 0;
                std::uint64_t distributed_weight = 0u;
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
                    chunk_stats = *stats;
                }
                if (!chunk_stats.has_value() || distributed_iterations != chunk_iterations) {
                    std::println("{}error:{} optimization iteration distribution failed.", ansi_red, ansi_reset);
                    return 1;
                }
                const hyfluid::train::OptimizationStats& stats = *chunk_stats;
                const hyfluid::inspector::TrainingBatchDiagnostics batch_diagnostics = inspector.training_batch_diagnostics();
                const hyfluid::inspector::TrainingModelDiagnostics model_diagnostics = inspector.training_model_diagnostics();

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

                std::println("density optimize: frame_set={} step={} loss={:.6g} psnr={:.3f} recent_psnr={:.3f} rays={}/{} samples={}/{} sample_eff={:.2f}% coord_min=[{:.4f},{:.4f},{:.4f}] coord_max=[{:.4f},{:.4f},{:.4f}] time=[{:.4f},{:.4f}] dt_metric=[{:.8f},{:.8f},{:.8f}] metric_per_field_unit=[{:.6f},{:.6f},{:.6f}] global_rgb=[param {:.6g}, color {:.6g}, grad {:.6g}] occupancy_cells={} occupancy_grid={:.2f}% elapsed={:.3f}ms",
                             optimize_frame_set_label,
                             stats.step,
                             stats.loss,
                             stats.psnr,
                             recent_psnr_mean,
                             stats.ray_count,
                             stats.rays_per_batch,
                             stats.sample_count,
                             stats.sample_count_before_compaction,
                             stats.sample_efficiency_ratio * 100.0f,
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
                             stats.occupancy_grid_ratio * 100.0f,
                             stats.elapsed_ms);

                if (training_log) {
                    training_log << std::format("{{\"step\":{},\"loss\":{:.9g},\"psnr\":{:.9g},\"recent_psnr\":{:.9g},\"ray_count\":{},\"rays_per_batch\":{},\"sample_count\":{},\"sample_count_before_compaction\":{},\"sample_efficiency_ratio\":{:.9g},\"coord_min\":[{:.9g},{:.9g},{:.9g}],\"coord_max\":[{:.9g},{:.9g},{:.9g}],\"time_min\":{:.9g},\"time_max\":{:.9g},\"dt_metric_min\":{:.9g},\"dt_metric_mean\":{:.9g},\"dt_metric_max\":{:.9g},\"metric_per_field_unit_min\":{:.9g},\"metric_per_field_unit_mean\":{:.9g},\"metric_per_field_unit_max\":{:.9g},\"field_metric_extent\":[{:.9g},{:.9g},{:.9g}],\"global_rgb_param\":{:.9g},\"global_rgb_color\":{:.9g},\"global_rgb_gradient\":{:.9g},\"occupancy_grid_occupied_cells\":{},\"occupancy_grid_ratio\":{:.9g},\"elapsed_ms\":{:.9g}}}\n",
                                                stats.step,
                                                stats.loss,
                                                stats.psnr,
                                                recent_psnr_mean,
                                                stats.ray_count,
                                                stats.rays_per_batch,
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
                                                training_domain.field_metric_extent[0u],
                                                training_domain.field_metric_extent[1u],
                                                training_domain.field_metric_extent[2u],
                                                model_diagnostics.global_rgb_param,
                                                model_diagnostics.global_rgb_color,
                                                model_diagnostics.global_rgb_gradient,
                                                stats.occupancy_grid_occupied_cells,
                                                stats.occupancy_grid_ratio,
                                                stats.elapsed_ms);
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
                std::println("density evaluate: frame_set={} step={} images={}/{} render={}x{} pixels={} mse={:.9g} psnr={:.3f} output={} elapsed={:.3f}ms",
                             evaluation->frame_set,
                             evaluation->step,
                             evaluation->rendered_image_count,
                             evaluation->image_count,
                             evaluation->render_width,
                             evaluation->render_height,
                             evaluation->pixel_count,
                             evaluation->mse,
                             evaluation->psnr,
                             evaluation->output_dir.string(),
                             evaluation->elapsed_ms);
            }
        }
        if (save_weights_path.has_value()) {
            const std::expected<void, std::string> saved_weights = hyfluid.export_weights(*save_weights_path);
            if (!saved_weights) {
                std::println("{}error:{} {}", ansi_red, ansi_reset, saved_weights.error());
                return 1;
            }
            std::println("weights saved: {}", save_weights_path->string());
        }

        return 0;
    } catch (const std::exception& error) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, error.what());
        return 1;
    }
}
