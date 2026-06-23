cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(HyFluid VERSION 1.2.0 LANGUAGES CXX CUDA)

option(HYFLUID_BUILD_APP "Build the optional HyFluid executable" OFF)

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

set(HYFLUID_TRAIN_PROFILE "baseline" CACHE STRING "Compile-time HyFluid train profile")
set_property(CACHE HYFLUID_TRAIN_PROFILE PROPERTY STRINGS baseline)
if (NOT HYFLUID_TRAIN_PROFILE STREQUAL "baseline")
    message(FATAL_ERROR "Unknown HYFLUID_TRAIN_PROFILE='${HYFLUID_TRAIN_PROFILE}'. Supported profiles: baseline")
endif ()
set(HYFLUID_TRAIN_PROFILE_DEFINITION_SUFFIX "BASELINE")

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

# ============================================================================
# Core Library
# ============================================================================
add_library(libhyfluid)
target_sources(
        libhyfluid
        PRIVATE
        hyfluid.train.cpp
        hyfluid.train.cu
        hyfluid.train.config.h
        hyfluid.train.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        hyfluid.train.ixx
)
target_compile_definitions(libhyfluid PRIVATE "HYFLUID_TRAIN_PROFILE_${HYFLUID_TRAIN_PROFILE_DEFINITION_SUFFIX}=1")
target_link_libraries(libhyfluid PUBLIC CUDA::cudart)
set_property(TARGET libhyfluid PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional Dataset Loading
# ============================================================================
if (HYFLUID_BUILD_APP)
    set(HYFLUID_DATASET_GIT_TAG "f4a4ff6ccd7f1c4f805d4deb2b616c0741e0273e")
    set(HYFLUID_DATASET_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/dataset/${HYFLUID_DATASET_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${HYFLUID_DATASET_GIT_TAG}/scalar_real.ixx"
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.ixx"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${HYFLUID_DATASET_GIT_TAG}/scalar_real.cpp"
            "${HYFLUID_DATASET_VENDOR_DIR}/scalar_real.cpp"
    )
    download_if_needed(
            "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
            "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
    )

    set(HYFLUID_FFMPEG_RELEASE_TAG "autobuild-2026-06-17-14-17")
    if (WIN32)
        set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1.2-win64-lgpl-shared-8.1")
        set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "zip")
        set(HYFLUID_FFMPEG_SHA256 "0ac5991693950b7f06483b19a4fa4752646d413d6535cb6310696de5baf7808f")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HYFLUID_SYSTEM_PROCESSOR)
        if (HYFLUID_SYSTEM_PROCESSOR STREQUAL "x86_64" OR HYFLUID_SYSTEM_PROCESSOR STREQUAL "amd64")
            set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1.2-linux64-lgpl-shared-8.1")
            set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "tar.xz")
            set(HYFLUID_FFMPEG_SHA256 "dd8d149f5f774638d64fb791f88e341d05407a8a4073cb8e137a6bc86f6d1d1e")
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

    download_if_needed("${HYFLUID_FFMPEG_URL}" "${HYFLUID_FFMPEG_ARCHIVE}")
    file(SHA256 "${HYFLUID_FFMPEG_ARCHIVE}" HYFLUID_FFMPEG_ARCHIVE_SHA256)
    if (NOT HYFLUID_FFMPEG_ARCHIVE_SHA256 STREQUAL HYFLUID_FFMPEG_SHA256)
        message(FATAL_ERROR "Vendor FFmpeg archive hash mismatch: ${HYFLUID_FFMPEG_ARCHIVE}")
    endif ()

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
# Optional App
# ============================================================================
if (HYFLUID_BUILD_APP)
    set(HYFLUID_UTIL_GIT_TAG "bb261075e7c8a4722b23f42c3be01aa24024731a")
    set(HYFLUID_UTIL_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/util/${HYFLUID_UTIL_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${HYFLUID_UTIL_GIT_TAG}/xcli.ixx"
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${HYFLUID_UTIL_GIT_TAG}/xcli.cpp"
            "${HYFLUID_UTIL_VENDOR_DIR}/xcli.cpp"
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
    target_compile_definitions(hyfluid PRIVATE "HYFLUID_TRAIN_PROFILE_NAME=\"${HYFLUID_TRAIN_PROFILE}\"")
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
endif ()
