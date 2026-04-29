cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(HyFluid VERSION 1.0.0 LANGUAGES CXX CUDA)


# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)

set(HYFLUID_FFMPEG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg")
set(HYFLUID_FFMPEG_INCLUDE_DIR "${HYFLUID_FFMPEG_ROOT}/include")
set(HYFLUID_FFMPEG_LIB_DIR "${HYFLUID_FFMPEG_ROOT}/lib")
set(HYFLUID_FFMPEG_BIN_DIR "${HYFLUID_FFMPEG_ROOT}/bin")
set(HYFLUID_FFMPEG_LIBRARIES
        "${HYFLUID_FFMPEG_LIB_DIR}/avformat.lib"
        "${HYFLUID_FFMPEG_LIB_DIR}/avcodec.lib"
        "${HYFLUID_FFMPEG_LIB_DIR}/swscale.lib"
        "${HYFLUID_FFMPEG_LIB_DIR}/swresample.lib"
        "${HYFLUID_FFMPEG_LIB_DIR}/avutil.lib"
)
set(HYFLUID_FFMPEG_RUNTIME_FILES
        "${HYFLUID_FFMPEG_BIN_DIR}/avformat-62.dll"
        "${HYFLUID_FFMPEG_BIN_DIR}/avcodec-62.dll"
        "${HYFLUID_FFMPEG_BIN_DIR}/swscale-9.dll"
        "${HYFLUID_FFMPEG_BIN_DIR}/swresample-6.dll"
        "${HYFLUID_FFMPEG_BIN_DIR}/avutil-60.dll"
)
if (NOT IS_DIRECTORY "${HYFLUID_FFMPEG_INCLUDE_DIR}")
    message(FATAL_ERROR "Repository FFmpeg include directory is missing: ${HYFLUID_FFMPEG_INCLUDE_DIR}")
endif ()
foreach (HYFLUID_FFMPEG_FILE IN LISTS HYFLUID_FFMPEG_LIBRARIES HYFLUID_FFMPEG_RUNTIME_FILES)
    if (NOT EXISTS "${HYFLUID_FFMPEG_FILE}")
        message(FATAL_ERROR "Repository FFmpeg file is missing: ${HYFLUID_FFMPEG_FILE}")
    endif ()
endforeach ()

function(download_if_needed url out_path)
    if (EXISTS "${out_path}")
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD
            "${url}"
            "${out_path}"
            TLS_VERIFY ON
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
endfunction()

download_if_needed(
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
)
download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image.h"
)
download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image_write.h"
)

# ============================================================================
# Libraries
# ============================================================================
add_library(hyfluid)
target_sources(
        hyfluid
        PRIVATE
        ngp.dataset.cpp
        ngp.train.cpp
        ngp.train.cu
        ngp.train.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ngp.dataset.ixx
        ngp.train.ixx
)
target_include_directories(hyfluid PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor" "${HYFLUID_FFMPEG_INCLUDE_DIR}")
target_link_libraries(hyfluid PRIVATE CUDA::cudart ${HYFLUID_FFMPEG_LIBRARIES})
set_property(TARGET hyfluid PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Executable
# ============================================================================
add_executable(hyfluid-app main.cpp)
target_link_libraries(hyfluid-app PRIVATE hyfluid)
add_custom_command(
        TARGET hyfluid-app POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${HYFLUID_FFMPEG_RUNTIME_FILES} "$<TARGET_FILE_DIR:hyfluid-app>"
)
set_property(TARGET hyfluid-app PROPERTY CXX_MODULE_STD ON)
