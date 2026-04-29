export module ngp.dataset;
import std;

namespace ngp::dataset {
    export struct ScalarRealFrame final {
        std::string file_name;
        std::vector<std::uint8_t> rgba;
        std::array<float, 12> camera = {};
        std::uint32_t width          = 0u;
        std::uint32_t height         = 0u;
        float focal_x                = 0.0f;
        float focal_y                = 0.0f;
        float principal_x            = 0.0f;
        float principal_y            = 0.0f;
        float time                   = 0.0f;
    };

    export struct ScalarRealVideo final {
        std::string file_name;
        std::vector<std::uint8_t> rgb;
        std::array<float, 12> camera = {};
        std::uint32_t width          = 0u;
        std::uint32_t height         = 0u;
        std::uint32_t frame_count    = 0u;
        std::uint32_t frame_rate     = 0u;
        float focal                  = 0.0f;
    };

    export struct ScalarRealDataset final {
        std::vector<ScalarRealFrame> train;
        std::vector<ScalarRealFrame> validation;
        std::vector<ScalarRealFrame> test;
        std::vector<ScalarRealVideo> train_videos;
        std::vector<ScalarRealVideo> test_videos;
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
} // namespace ngp::dataset
