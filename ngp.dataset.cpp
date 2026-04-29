module;
#include "hyfluid.ffmpeg.h"
#include <cstring>

#include "json/json.hpp"
module hyfluid.dataset;
import std;

namespace hyfluid::dataset {
    std::expected<ScalarRealDataset, std::string> load_scalar_real(const std::filesystem::path& path) {
        try {
            if (path.empty()) throw std::runtime_error{"dataset path must not be empty."};
            if (!std::filesystem::is_directory(path)) throw std::runtime_error{std::format("dataset path '{}' is not a directory.", path.string())};

            const std::filesystem::path info_path = path / "info.json";
            if (!std::filesystem::is_regular_file(info_path)) throw std::runtime_error{std::format("ScalarReal dataset '{}' is missing info.json.", path.string())};

            const nlohmann::json json            = nlohmann::json::parse(std::ifstream{info_path, std::ios::binary}, nullptr, true, true);
            ScalarRealDataset dataset            = {};
            dataset.near                         = json.at("near").get<float>();
            dataset.far                          = json.at("far").get<float>();
            dataset.phi                          = json.at("phi").get<float>();
            const std::string rotation_axis_text = json.at("rot").get<std::string>();
            if (rotation_axis_text.size() != 1 || (rotation_axis_text[0] != 'Y' && rotation_axis_text[0] != 'Z')) throw std::runtime_error{"ScalarReal info.json rot must be Y or Z."};
            dataset.rotation_axis = rotation_axis_text[0];
            if (!std::isfinite(dataset.near) || !std::isfinite(dataset.far) || dataset.near <= 0.0f || dataset.far <= dataset.near) throw std::runtime_error{"ScalarReal info.json declares invalid near/far."};

            const nlohmann::json& render_center_json = json.at("render_center");
            for (std::size_t i = 0; i < 3; ++i) dataset.render_center[i] = render_center_json.at(i).get<float>();

            const nlohmann::json& voxel_scale_json = json.at("voxel_scale");
            for (std::size_t i = 0; i < 3; ++i) {
                dataset.voxel_scale[i] = voxel_scale_json.at(i).get<float>();
                if (!std::isfinite(dataset.voxel_scale[i]) || dataset.voxel_scale[i] == 0.0f) throw std::runtime_error{"ScalarReal info.json declares invalid voxel_scale."};
            }

            std::array<float, 16> voxel_matrix = {};
            const nlohmann::json& voxel_json   = json.at("voxel_matrix");
            for (std::size_t row = 0; row < 4; ++row) {
                voxel_matrix[row * 4 + 0] = voxel_json.at(row).at(2).get<float>();
                voxel_matrix[row * 4 + 1] = voxel_json.at(row).at(1).get<float>();
                voxel_matrix[row * 4 + 2] = voxel_json.at(row).at(0).get<float>();
                voxel_matrix[row * 4 + 3] = voxel_json.at(row).at(3).get<float>();
            }
            dataset.sim_to_world = voxel_matrix;

            std::array<float, 32> inverse_work = {};
            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t column = 0; column < 4; ++column) inverse_work[row * 8 + column] = voxel_matrix[row * 4 + column];
                inverse_work[row * 8 + 4 + row] = 1.0f;
            }
            for (std::size_t pivot_column = 0; pivot_column < 4; ++pivot_column) {
                std::size_t pivot_row = pivot_column;
                float pivot_abs       = std::fabs(inverse_work[pivot_row * 8 + pivot_column]);
                for (std::size_t row = pivot_column + 1; row < 4; ++row) {
                    const float candidate_abs = std::fabs(inverse_work[row * 8 + pivot_column]);
                    if (candidate_abs > pivot_abs) {
                        pivot_abs = candidate_abs;
                        pivot_row = row;
                    }
                }
                if (pivot_abs <= 1e-12f) throw std::runtime_error{"ScalarReal voxel_matrix is singular."};
                if (pivot_row != pivot_column) {
                    for (std::size_t column = 0; column < 8; ++column) std::swap(inverse_work[pivot_row * 8 + column], inverse_work[pivot_column * 8 + column]);
                }
                const float pivot = inverse_work[pivot_column * 8 + pivot_column];
                for (std::size_t column = 0; column < 8; ++column) inverse_work[pivot_column * 8 + column] /= pivot;
                for (std::size_t row = 0; row < 4; ++row) {
                    if (row == pivot_column) continue;
                    const float factor = inverse_work[row * 8 + pivot_column];
                    for (std::size_t column = 0; column < 8; ++column) inverse_work[row * 8 + column] -= factor * inverse_work[pivot_column * 8 + column];
                }
            }
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t column = 0; column < 4; ++column) dataset.world_to_sim[row * 4 + column] = inverse_work[row * 8 + 4 + column];

            const std::array<std::string_view, 2> split_names = {"train_videos", "test_videos"};
            for (const std::size_t split_index : std::views::iota(std::size_t{0}, split_names.size())) {
                const nlohmann::json& videos_json = json.at(split_names[split_index]);
                if (!videos_json.is_array() || videos_json.empty()) throw std::runtime_error{std::format("ScalarReal info.json must contain a non-empty {} array.", split_names[split_index])};

                std::vector<ScalarRealVideo>& destination = split_index == 0 ? dataset.train : dataset.test;
                destination.reserve(videos_json.size());

                for (const nlohmann::json& video_json : videos_json) {
                    ScalarRealVideo video      = {};
                    video.file_name            = video_json.at("file_name").get<std::string>();
                    video.frame_count          = video_json.at("frame_num").get<std::uint32_t>();
                    video.frame_rate           = video_json.at("frame_rate").get<std::uint32_t>();
                    const float camera_angle_x = video_json.at("camera_angle_x").get<float>();
                    if (video.file_name.empty()) throw std::runtime_error{std::format("{} contains an empty file_name.", split_names[split_index])};
                    if (video.frame_count == 0u) throw std::runtime_error{std::format("{} declares zero frame_num.", video.file_name)};
                    if (video.frame_rate == 0u) throw std::runtime_error{std::format("{} declares zero frame_rate.", video.file_name)};
                    if (!std::isfinite(camera_angle_x) || camera_angle_x <= 0.0f) throw std::runtime_error{std::format("{} declares invalid camera_angle_x.", video.file_name)};
                    if (video_json.contains("transform_matrix_list")) throw std::runtime_error{std::format("{} uses transform_matrix_list; this HyFluid density implementation expects one static transform_matrix per video.", video.file_name)};

                    const nlohmann::json& camera_json = video_json.at("transform_matrix");
                    for (std::size_t row = 0; row < 4; ++row)
                        for (std::size_t column = 0; column < 4; ++column) video.camera[row * 4 + column] = camera_json.at(row).at(column).get<float>();

                    AVFormatContext* format_context        = nullptr;
                    const std::filesystem::path video_path = path / video.file_name;
                    if (!std::filesystem::is_regular_file(video_path)) throw std::runtime_error{std::format("ScalarReal video '{}' does not exist.", video_path.string())};
                    if (avformat_open_input(&format_context, video_path.string().c_str(), nullptr, nullptr) < 0) throw std::runtime_error{std::format("failed to open video '{}'.", video_path.string())};

                    AVCodecContext* codec_context = nullptr;
                    SwsContext* sws_context       = nullptr;
                    AVPacket* packet              = nullptr;
                    AVFrame* frame                = nullptr;
                    AVFrame* rgb_frame            = nullptr;
                    std::uint8_t* rgb_buffer      = nullptr;
                    try {
                        if (avformat_find_stream_info(format_context, nullptr) < 0) throw std::runtime_error{std::format("failed to read stream info from '{}'.", video_path.string())};
                        int video_stream_index = -1;
                        for (unsigned int stream_index = 0u; stream_index < format_context->nb_streams; ++stream_index) {
                            if (format_context->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                                video_stream_index = static_cast<int>(stream_index);
                                break;
                            }
                        }
                        if (video_stream_index < 0) throw std::runtime_error{std::format("'{}' contains no video stream.", video_path.string())};

                        AVStream* stream       = format_context->streams[video_stream_index];
                        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
                        if (decoder == nullptr) throw std::runtime_error{std::format("no FFmpeg decoder is available for '{}'.", video_path.string())};
                        codec_context = avcodec_alloc_context3(decoder);
                        if (codec_context == nullptr) throw std::runtime_error{"avcodec_alloc_context3 failed."};
                        if (avcodec_parameters_to_context(codec_context, stream->codecpar) < 0) throw std::runtime_error{std::format("failed to copy codec parameters for '{}'.", video_path.string())};
                        if (avcodec_open2(codec_context, decoder, nullptr) < 0) throw std::runtime_error{std::format("failed to open decoder for '{}'.", video_path.string())};

                        if (codec_context->width <= 0 || codec_context->height <= 0) throw std::runtime_error{std::format("'{}' has invalid dimensions.", video_path.string())};
                        video.width  = static_cast<std::uint32_t>(codec_context->width);
                        video.height = static_cast<std::uint32_t>(codec_context->height);
                        video.focal  = 0.5f * static_cast<float>(video.width) / std::tan(0.5f * camera_angle_x);
                        if (!std::isfinite(video.focal) || video.focal <= 0.0f) throw std::runtime_error{std::format("{} produced invalid focal length.", video.file_name)};

                        if (stream->avg_frame_rate.den == 0 || stream->avg_frame_rate.num == 0) throw std::runtime_error{std::format("'{}' has no average frame rate.", video_path.string())};
                        const std::uint32_t probed_frame_rate = static_cast<std::uint32_t>(std::llround(static_cast<double>(stream->avg_frame_rate.num) / static_cast<double>(stream->avg_frame_rate.den)));
                        if (probed_frame_rate != video.frame_rate) throw std::runtime_error{std::format("{} declares {} fps but FFmpeg reports {} fps.", video.file_name, video.frame_rate, probed_frame_rate)};

                        sws_context = sws_getContext(codec_context->width, codec_context->height, codec_context->pix_fmt, codec_context->width, codec_context->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
                        if (sws_context == nullptr) throw std::runtime_error{std::format("sws_getContext failed for '{}'.", video_path.string())};

                        packet    = av_packet_alloc();
                        frame     = av_frame_alloc();
                        rgb_frame = av_frame_alloc();
                        if (packet == nullptr || frame == nullptr || rgb_frame == nullptr) throw std::runtime_error{"FFmpeg allocation failed."};
                        if (av_image_alloc(rgb_frame->data, rgb_frame->linesize, codec_context->width, codec_context->height, AV_PIX_FMT_RGB24, 32) < 0) throw std::runtime_error{std::format("av_image_alloc failed for '{}'.", video_path.string())};
                        rgb_buffer = rgb_frame->data[0];

                        const std::uint64_t video_byte_count = static_cast<std::uint64_t>(video.frame_count) * video.width * video.height * 3ull;
                        if (video_byte_count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{std::format("{} is too large to store on this host.", video.file_name)};
                        video.rgb.resize(static_cast<std::size_t>(video_byte_count));
                        std::uint32_t decoded_frame_index = 0u;

                        while (av_read_frame(format_context, packet) >= 0) {
                            if (packet->stream_index == video_stream_index) {
                                if (avcodec_send_packet(codec_context, packet) < 0) throw std::runtime_error{std::format("avcodec_send_packet failed for '{}'.", video_path.string())};
                                while (true) {
                                    const int receive_status = avcodec_receive_frame(codec_context, frame);
                                    if (receive_status == AVERROR(EAGAIN) || receive_status == AVERROR_EOF) break;
                                    if (receive_status < 0) throw std::runtime_error{std::format("avcodec_receive_frame failed for '{}'.", video_path.string())};
                                    if (decoded_frame_index >= video.frame_count) throw std::runtime_error{std::format("{} decoded more than {} frames.", video.file_name, video.frame_count)};
                                    sws_scale(sws_context, frame->data, frame->linesize, 0, codec_context->height, rgb_frame->data, rgb_frame->linesize);
                                    for (std::uint32_t row = 0u; row < video.height; ++row) {
                                        const std::uint8_t* row_begin = rgb_frame->data[0] + static_cast<std::ptrdiff_t>(row) * rgb_frame->linesize[0];
                                        std::uint8_t* row_out         = video.rgb.data() + (static_cast<std::uint64_t>(decoded_frame_index) * video.height + row) * static_cast<std::uint64_t>(video.width) * 3ull;
                                        std::memcpy(row_out, row_begin, static_cast<std::size_t>(video.width) * 3);
                                    }
                                    ++decoded_frame_index;
                                }
                            }
                            av_packet_unref(packet);
                        }

                        if (avcodec_send_packet(codec_context, nullptr) < 0) throw std::runtime_error{std::format("failed to flush decoder for '{}'.", video_path.string())};
                        while (true) {
                            const int receive_status = avcodec_receive_frame(codec_context, frame);
                            if (receive_status == AVERROR_EOF || receive_status == AVERROR(EAGAIN)) break;
                            if (receive_status < 0) throw std::runtime_error{std::format("avcodec_receive_frame flush failed for '{}'.", video_path.string())};
                            if (decoded_frame_index >= video.frame_count) throw std::runtime_error{std::format("{} decoded more than {} frames.", video.file_name, video.frame_count)};
                            sws_scale(sws_context, frame->data, frame->linesize, 0, codec_context->height, rgb_frame->data, rgb_frame->linesize);
                            for (std::uint32_t row = 0u; row < video.height; ++row) {
                                const std::uint8_t* row_begin = rgb_frame->data[0] + static_cast<std::ptrdiff_t>(row) * rgb_frame->linesize[0];
                                std::uint8_t* row_out         = video.rgb.data() + (static_cast<std::uint64_t>(decoded_frame_index) * video.height + row) * static_cast<std::uint64_t>(video.width) * 3ull;
                                std::memcpy(row_out, row_begin, static_cast<std::size_t>(video.width) * 3);
                            }
                            ++decoded_frame_index;
                        }

                        if (decoded_frame_index != video.frame_count) throw std::runtime_error{std::format("{} declares {} frames but FFmpeg decoded {} frames.", video.file_name, video.frame_count, decoded_frame_index)};
                    } catch (...) {
                        if (rgb_buffer != nullptr) av_freep(&rgb_frame->data[0]);
                        if (rgb_frame != nullptr) av_frame_free(&rgb_frame);
                        if (frame != nullptr) av_frame_free(&frame);
                        if (packet != nullptr) av_packet_free(&packet);
                        if (sws_context != nullptr) sws_freeContext(sws_context);
                        if (codec_context != nullptr) avcodec_free_context(&codec_context);
                        if (format_context != nullptr) avformat_close_input(&format_context);
                        throw;
                    }

                    if (rgb_buffer != nullptr) av_freep(&rgb_frame->data[0]);
                    av_frame_free(&rgb_frame);
                    av_frame_free(&frame);
                    av_packet_free(&packet);
                    sws_freeContext(sws_context);
                    avcodec_free_context(&codec_context);
                    avformat_close_input(&format_context);

                    if (!destination.empty()) {
                        const ScalarRealVideo& first = destination.front();
                        if (video.width != first.width || video.height != first.height) throw std::runtime_error{"ScalarReal videos must share one resolution."};
                        if (video.frame_count != first.frame_count) throw std::runtime_error{"ScalarReal videos must share one frame count."};
                    }
                    destination.push_back(std::move(video));
                }
            }

            if (dataset.train.empty()) throw std::runtime_error{"ScalarReal train_videos is empty."};
            if (dataset.test.empty()) throw std::runtime_error{"ScalarReal test_videos is empty."};
            return dataset;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace hyfluid::dataset
