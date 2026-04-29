export module hyfluid.dataset;
import std;

namespace hyfluid::dataset {
    export struct ScalarRealVideo final {
        std::string file_name;
        std::vector<std::uint8_t> rgb;
        std::array<float, 16> camera = {};
        std::uint32_t width          = 0u;
        std::uint32_t height         = 0u;
        std::uint32_t frame_count    = 0u;
        std::uint32_t frame_rate     = 0u;
        float focal                  = 0.0f;
    };

    export struct ScalarRealDataset final {
        std::vector<ScalarRealVideo> train;
        std::vector<ScalarRealVideo> test;
        std::array<float, 16> sim_to_world = {};
        std::array<float, 16> world_to_sim = {};
        std::array<float, 3> voxel_scale   = {};
        std::array<float, 3> render_center = {};
        float near                         = 0.0f;
        float far                          = 0.0f;
        float phi                          = 0.0f;
        char rotation_axis                 = 'Y';
    };

    export std::expected<ScalarRealDataset, std::string> load_scalar_real(const std::filesystem::path& path);
} // namespace hyfluid::dataset
