export module hyfluid.project;

import hyfluid.plugin;
import std;

export namespace hyfluid::project {
    class Project final {
    public:
        struct State;

        Project(const Project& other) = delete;
        Project(Project&& other) noexcept;
        Project& operator=(const Project& other) = delete;
        Project& operator=(Project&& other) noexcept;
        ~Project() noexcept;

        [[nodiscard]] static const plugin::PluginDefinition<Project>& plugin();
        [[nodiscard]] static Project open(plugin::OpenContext context);

        void update(const plugin::UpdateInfo& update);
        [[nodiscard]] std::uint64_t revision() const;
        [[nodiscard]] plugin::Document document() const;
        [[nodiscard]] plugin::Frame frame(const plugin::FrameInfo& frame) const;
        void write_controls(plugin::ControlBuilder& controls) const;

    private:
        explicit Project(std::unique_ptr<State> state);

        std::unique_ptr<State> state;
    };
} // namespace hyfluid::project
