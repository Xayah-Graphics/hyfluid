#ifndef HYFLUID_TRAIN_CONFIG_H
#define HYFLUID_TRAIN_CONFIG_H

#include <cstdint>
#include <string_view>

namespace hyfluid::train::config {
#if !defined(HYFLUID_TRAIN_PROFILE_BASELINE)
#error "HYFLUID_TRAIN_PROFILE_BASELINE must be selected."
#endif

    inline constexpr std::string_view active_profile_name = "baseline";
    inline constexpr std::uint32_t rgba_channel_count = 4u;
    inline constexpr std::uint32_t camera_value_count = 12u;
} // namespace hyfluid::train::config

#endif // HYFLUID_TRAIN_CONFIG_H
