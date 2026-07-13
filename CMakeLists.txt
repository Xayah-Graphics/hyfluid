cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(HyFluid VERSION 1.1.0 LANGUAGES CXX CUDA)

option(HYFLUID_BUILD_APP "Build the optional HyFluid executable" OFF)
option(HYFLUID_BUILD_PROJECT_PLUGIN "Build the optional Spectra project plugin" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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

function(download_if_needed url out_path sha256)
    string(TOLOWER "${sha256}" expected_sha256)
    if (EXISTS "${out_path}")
        file(SHA256 "${out_path}" existing_sha256)
        string(TOLOWER "${existing_sha256}" existing_sha256)
        if (NOT existing_sha256 STREQUAL expected_sha256)
            message(FATAL_ERROR "Cached file hash mismatch for ${out_path}: expected ${expected_sha256}, got ${existing_sha256}")
        endif ()
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    set(tmp_path "${out_path}.download")
    file(REMOVE "${tmp_path}")
    file(DOWNLOAD
            "${url}"
            "${tmp_path}"
            TLS_VERIFY ON
            EXPECTED_HASH "SHA256=${sha256}"
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        file(REMOVE "${tmp_path}")
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
    file(RENAME "${tmp_path}" "${out_path}")
endfunction()

download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image_write.h"
        "CBD5F0AD7A9CF4468AFFB36354A1D2338034F2C12473CF1A8E32053CB6914A05"
)
download_if_needed(
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
        "AAF127C04CB31C406E5B04A63F1AE89369FCCDE6D8FA7CDDA1ED4F32DFC5DE63"
)

# ============================================================================
# Core Library
# ============================================================================
add_library(libhyfluid)
target_sources(
        libhyfluid
        PRIVATE
        hyfluid.train.cpp
        hyfluid.train.cu
        hyfluid.train.h
        hyfluid.train.config.h
        hyfluid.inspector.cpp
        hyfluid.inspector.cu
        hyfluid.inspector.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        hyfluid.train.ixx
        hyfluid.inspector.ixx
)
target_link_libraries(libhyfluid PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_include_directories(libhyfluid PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
set_property(TARGET libhyfluid PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional Dataset Loading
# ============================================================================
if (HYFLUID_BUILD_APP OR HYFLUID_BUILD_PROJECT_PLUGIN)
    set(HYFLUID_DATASET_GIT_TAG "a91368d236fe236779f4a816ff7eee25dec127cb")
    set(HYFLUID_DATASET_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/dataset/${HYFLUID_DATASET_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${HYFLUID_DATASET_GIT_TAG}/scalar_real.ixx"
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.ixx"
            "9E6510216C3753EF534850D8F0F6AADDE72BD6D987669888CDA62E972B16835F"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${HYFLUID_DATASET_GIT_TAG}/scalar_real.cpp"
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.cpp"
            "1FBABA027DFC45DF5864AE07E30B8901635738D1EE0EA115E32341411B29C20C"
    )
    # BtbN retains month-end autobuilds longer than ordinary daily autobuilds.
    set(HYFLUID_FFMPEG_RELEASE_TAG "autobuild-2026-06-30-13-34")
    if (WIN32)
        set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1.2-21-gce3c09c101-win64-lgpl-shared-8.1")
        set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "zip")
        set(HYFLUID_FFMPEG_SHA256 "27bcaf58b5140171dfe838a0b365d12c60607d71fc168424456410bad6a834da")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HYFLUID_SYSTEM_PROCESSOR)
        if (HYFLUID_SYSTEM_PROCESSOR STREQUAL "x86_64" OR HYFLUID_SYSTEM_PROCESSOR STREQUAL "amd64")
            set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1.2-21-gce3c09c101-linux64-lgpl-shared-8.1")
            set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "tar.xz")
            set(HYFLUID_FFMPEG_SHA256 "bc60572d89533c7987e7cfbb654eba4d35122a77885c6938483dd7ef8219cde9")
        else ()
            message(FATAL_ERROR "Unsupported Linux processor for vendor FFmpeg: ${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
    else ()
        message(FATAL_ERROR "Unsupported platform for vendor FFmpeg: ${CMAKE_SYSTEM_NAME}")
    endif ()
    set(HYFLUID_FFMPEG_ARCHIVE_NAME "${HYFLUID_FFMPEG_NAME}.${HYFLUID_FFMPEG_ARCHIVE_EXTENSION}")
    set(HYFLUID_FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/${HYFLUID_FFMPEG_RELEASE_TAG}/${HYFLUID_FFMPEG_ARCHIVE_NAME}")
    set(HYFLUID_FFMPEG_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/ffmpeg")
    set(HYFLUID_FFMPEG_ARCHIVE "${HYFLUID_FFMPEG_VENDOR_DIR}/${HYFLUID_FFMPEG_ARCHIVE_NAME}")
    set(HYFLUID_FFMPEG_EXTRACT_DIR "${HYFLUID_FFMPEG_VENDOR_DIR}/extract")
    set(HYFLUID_FFMPEG_MARKER "${HYFLUID_FFMPEG_VENDOR_DIR}/${HYFLUID_FFMPEG_ARCHIVE_NAME}.sha256")
    set(HYFLUID_FFMPEG_ROOT "${HYFLUID_FFMPEG_EXTRACT_DIR}/${HYFLUID_FFMPEG_NAME}")
    set(HYFLUID_FFMPEG_INCLUDE_DIR "${HYFLUID_FFMPEG_ROOT}/include")
    set(HYFLUID_FFMPEG_LIB_DIR "${HYFLUID_FFMPEG_ROOT}/lib")
    set(HYFLUID_FFMPEG_BIN_DIR "${HYFLUID_FFMPEG_ROOT}/bin")
    if (WIN32)
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
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(HYFLUID_FFMPEG_LIBRARIES
                "${HYFLUID_FFMPEG_LIB_DIR}/libavformat.so"
                "${HYFLUID_FFMPEG_LIB_DIR}/libavcodec.so"
                "${HYFLUID_FFMPEG_LIB_DIR}/libswscale.so"
                "${HYFLUID_FFMPEG_LIB_DIR}/libswresample.so"
                "${HYFLUID_FFMPEG_LIB_DIR}/libavutil.so"
        )
    endif ()

    string(LENGTH "${HYFLUID_FFMPEG_SHA256}" HYFLUID_FFMPEG_SHA256_LENGTH)
    if (NOT HYFLUID_FFMPEG_SHA256_LENGTH EQUAL 64)
        message(FATAL_ERROR "Invalid configured SHA256 for ${HYFLUID_FFMPEG_ARCHIVE_NAME}")
    endif ()

    download_if_needed("${HYFLUID_FFMPEG_URL}" "${HYFLUID_FFMPEG_ARCHIVE}" "${HYFLUID_FFMPEG_SHA256}")

    set(HYFLUID_FFMPEG_EXISTING_SHA256 "")
    if (EXISTS "${HYFLUID_FFMPEG_MARKER}")
        file(READ "${HYFLUID_FFMPEG_MARKER}" HYFLUID_FFMPEG_EXISTING_SHA256)
        string(STRIP "${HYFLUID_FFMPEG_EXISTING_SHA256}" HYFLUID_FFMPEG_EXISTING_SHA256)
    endif ()
    if ((NOT HYFLUID_FFMPEG_EXISTING_SHA256 STREQUAL HYFLUID_FFMPEG_SHA256) OR (NOT IS_DIRECTORY "${HYFLUID_FFMPEG_ROOT}"))
        message(STATUS "Extracting ${HYFLUID_FFMPEG_ARCHIVE}")
        file(REMOVE_RECURSE "${HYFLUID_FFMPEG_EXTRACT_DIR}")
        file(MAKE_DIRECTORY "${HYFLUID_FFMPEG_EXTRACT_DIR}")
        file(ARCHIVE_EXTRACT INPUT "${HYFLUID_FFMPEG_ARCHIVE}" DESTINATION "${HYFLUID_FFMPEG_EXTRACT_DIR}")
        file(WRITE "${HYFLUID_FFMPEG_MARKER}" "${HYFLUID_FFMPEG_SHA256}\n")
    endif ()

    if (NOT IS_DIRECTORY "${HYFLUID_FFMPEG_INCLUDE_DIR}")
        message(FATAL_ERROR "Vendor FFmpeg include directory is missing: ${HYFLUID_FFMPEG_INCLUDE_DIR}")
    endif ()
    foreach (HYFLUID_FFMPEG_FILE IN LISTS HYFLUID_FFMPEG_LIBRARIES HYFLUID_FFMPEG_RUNTIME_FILES)
        if (NOT EXISTS "${HYFLUID_FFMPEG_FILE}")
            message(FATAL_ERROR "Vendor FFmpeg file is missing: ${HYFLUID_FFMPEG_FILE}")
        endif ()
    endforeach ()

    add_library(dataset-scalar-real)
    target_sources(
            dataset-scalar-real
            PRIVATE
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.ixx"
    )
    target_include_directories(dataset-scalar-real PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor" "${HYFLUID_FFMPEG_INCLUDE_DIR}")
    target_link_libraries(dataset-scalar-real PRIVATE ${HYFLUID_FFMPEG_LIBRARIES})
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_property(TARGET dataset-scalar-real PROPERTY BUILD_RPATH "${HYFLUID_FFMPEG_LIB_DIR}")
    endif ()
    set_property(TARGET dataset-scalar-real PROPERTY CXX_MODULE_STD ON)
endif ()


# ============================================================================
# Optional Project Plugin
# ============================================================================
if (HYFLUID_BUILD_PROJECT_PLUGIN)
    add_library(hyfluid_project_plugin SHARED)
    target_sources(
            hyfluid_project_plugin
            PRIVATE
            hyfluid.project.cpp
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            hyfluid.plugin.ixx
            hyfluid.project.ixx
    )
    target_link_libraries(hyfluid_project_plugin PRIVATE dataset-scalar-real libhyfluid CUDA::cudart)
    if (WIN32)
        add_custom_command(
                TARGET hyfluid_project_plugin POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HYFLUID_FFMPEG_RUNTIME_FILES} $<TARGET_FILE_DIR:hyfluid_project_plugin>
        )
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_property(TARGET hyfluid_project_plugin PROPERTY BUILD_RPATH "${HYFLUID_FFMPEG_LIB_DIR}")
    endif ()
    set_property(TARGET hyfluid_project_plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(hyfluid_project_plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()


# ============================================================================
# Optional App
# ============================================================================
if (HYFLUID_BUILD_APP)
    set(HYFLUID_UTIL_GIT_TAG "bb261075e7c8a4722b23f42c3be01aa24024731a")
    set(HYFLUID_UTIL_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/util/${HYFLUID_UTIL_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${HYFLUID_UTIL_GIT_TAG}/xcli.ixx"
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.ixx"
            "2F80115AA7B739BDE4BF990EC1C0D0AFD85CF6B5610AF9A9D22A8C6686898F8D"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${HYFLUID_UTIL_GIT_TAG}/xcli.cpp"
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.cpp"
            "A1386CE6782CF267CD9AFB002FA66E70B7BF019CFA19ADC9C5EF0470BE445E69"
    )

    add_library(xayah-util-xcli)
    target_sources(
            xayah-util-xcli
            PRIVATE
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    set_property(TARGET xayah-util-xcli PROPERTY CXX_MODULE_STD ON)

    add_executable(hyfluid main.cpp)
    target_link_libraries(hyfluid PRIVATE xayah-util-xcli dataset-scalar-real libhyfluid)
    if (WIN32)
        add_custom_command(
                TARGET hyfluid POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HYFLUID_FFMPEG_RUNTIME_FILES} $<TARGET_FILE_DIR:hyfluid>
        )
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_property(TARGET hyfluid PROPERTY BUILD_RPATH "${HYFLUID_FFMPEG_LIB_DIR}")
    endif ()
    set_property(TARGET hyfluid PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(hyfluid PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()
