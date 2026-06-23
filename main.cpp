import std;
import xayah.util.xcli;

#ifndef HYFLUID_TRAIN_PROFILE_NAME
#error "HYFLUID_TRAIN_PROFILE_NAME must be provided by the app target."
#endif

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    constexpr std::string_view ansi_reset = "\x1b[0m";
    constexpr std::string_view ansi_red = "\x1b[31m";
    constexpr std::string_view active_train_profile_name = HYFLUID_TRAIN_PROFILE_NAME;

    xayah::util::Command command =
        xayah::util::Command{"HyFluid clean training framework entry."}
        | xayah::util::example("--help");

    const std::string usage = command.help(arguments);
    const auto cli_result = command.parse(arguments);
    if (!cli_result) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, cli_result.error());
        std::println("{}", usage);
        return 2;
    }
    if (cli_result->help_requested) {
        std::println("{}", usage);
        return 0;
    }

    std::println("{}error:{} HyFluid profile '{}' is a clean framework skeleton; training workflow is not implemented yet.", ansi_red, ansi_reset, active_train_profile_name);
    return 1;
}
