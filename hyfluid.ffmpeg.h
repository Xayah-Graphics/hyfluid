#ifndef HYFLUID_FFMPEG_H
#define HYFLUID_FFMPEG_H

#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#endif // HYFLUID_FFMPEG_H
