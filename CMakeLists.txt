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

set(HYFLUID_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor")
set(HYFLUID_FFMPEG_RELEASE_TAG "autobuild-2026-04-28-15-54")
if (WIN32)
    set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1-10-g7f5c90f77e-win64-lgpl-shared-8.1")
    set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "zip")
    set(HYFLUID_FFMPEG_SHA256 "f700bcb11b1b5fcc1c49a8b16d56d2aa4ec42099b4e1feb657b4dcf1a7f02ee3")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HYFLUID_SYSTEM_PROCESSOR)
    if (HYFLUID_SYSTEM_PROCESSOR STREQUAL "x86_64" OR HYFLUID_SYSTEM_PROCESSOR STREQUAL "amd64")
        set(HYFLUID_FFMPEG_NAME "ffmpeg-n8.1-10-g7f5c90f77e-linux64-lgpl-shared-8.1")
        set(HYFLUID_FFMPEG_ARCHIVE_EXTENSION "tar.xz")
        set(HYFLUID_FFMPEG_SHA256 "4ba68d323e8d767a285ce49f417f08ea0567ab6761b27962c20f818d8ace7acc")
    else ()
        message(FATAL_ERROR "Unsupported Linux processor for vendor FFmpeg: ${CMAKE_SYSTEM_PROCESSOR}")
    endif ()
else ()
    message(FATAL_ERROR "Unsupported platform for vendor FFmpeg: ${CMAKE_SYSTEM_NAME}")
endif ()
set(HYFLUID_FFMPEG_ARCHIVE_NAME "${HYFLUID_FFMPEG_NAME}.${HYFLUID_FFMPEG_ARCHIVE_EXTENSION}")
set(HYFLUID_FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/${HYFLUID_FFMPEG_RELEASE_TAG}/${HYFLUID_FFMPEG_ARCHIVE_NAME}")
set(HYFLUID_FFMPEG_VENDOR_DIR "${HYFLUID_VENDOR_DIR}/ffmpeg")
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

file(MAKE_DIRECTORY "${HYFLUID_FFMPEG_VENDOR_DIR}")
message(STATUS "Downloading ${HYFLUID_FFMPEG_URL}")
file(DOWNLOAD
        "${HYFLUID_FFMPEG_URL}"
        "${HYFLUID_FFMPEG_ARCHIVE}"
        EXPECTED_HASH "SHA256=${HYFLUID_FFMPEG_SHA256}"
        TLS_VERIFY ON
        STATUS HYFLUID_DOWNLOAD_STATUS
)
list(GET HYFLUID_DOWNLOAD_STATUS 0 HYFLUID_DOWNLOAD_CODE)
list(GET HYFLUID_DOWNLOAD_STATUS 1 HYFLUID_DOWNLOAD_MESSAGE)
if (NOT HYFLUID_DOWNLOAD_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download ${HYFLUID_FFMPEG_URL}: [${HYFLUID_DOWNLOAD_CODE}] ${HYFLUID_DOWNLOAD_MESSAGE}")
endif ()

set(HYFLUID_FFMPEG_EXISTING_SHA256 "")
if (EXISTS "${HYFLUID_FFMPEG_MARKER}")
    file(READ "${HYFLUID_FFMPEG_MARKER}" HYFLUID_FFMPEG_EXISTING_SHA256)
    string(STRIP "${HYFLUID_FFMPEG_EXISTING_SHA256}" HYFLUID_FFMPEG_EXISTING_SHA256)
endif ()
if (NOT HYFLUID_FFMPEG_EXISTING_SHA256 STREQUAL HYFLUID_FFMPEG_SHA256)
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

set(HYFLUID_JSON_HEADER "${HYFLUID_VENDOR_DIR}/json/json.hpp")
if (EXISTS "${HYFLUID_JSON_HEADER}")
    message(STATUS "Using cached file: ${HYFLUID_JSON_HEADER}")
else ()
    file(MAKE_DIRECTORY "${HYFLUID_VENDOR_DIR}/json")
    message(STATUS "Downloading https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp")
    file(DOWNLOAD
            "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
            "${HYFLUID_JSON_HEADER}"
            TLS_VERIFY ON
            STATUS HYFLUID_DOWNLOAD_STATUS
    )
    list(GET HYFLUID_DOWNLOAD_STATUS 0 HYFLUID_DOWNLOAD_CODE)
    list(GET HYFLUID_DOWNLOAD_STATUS 1 HYFLUID_DOWNLOAD_MESSAGE)
    if (NOT HYFLUID_DOWNLOAD_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp: [${HYFLUID_DOWNLOAD_CODE}] ${HYFLUID_DOWNLOAD_MESSAGE}")
    endif ()
endif ()

set(HYFLUID_STB_IMAGE_HEADER "${HYFLUID_VENDOR_DIR}/stb/stb_image.h")
if (EXISTS "${HYFLUID_STB_IMAGE_HEADER}")
    message(STATUS "Using cached file: ${HYFLUID_STB_IMAGE_HEADER}")
else ()
    file(MAKE_DIRECTORY "${HYFLUID_VENDOR_DIR}/stb")
    message(STATUS "Downloading https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h")
    file(DOWNLOAD
            "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
            "${HYFLUID_STB_IMAGE_HEADER}"
            TLS_VERIFY ON
            STATUS HYFLUID_DOWNLOAD_STATUS
    )
    list(GET HYFLUID_DOWNLOAD_STATUS 0 HYFLUID_DOWNLOAD_CODE)
    list(GET HYFLUID_DOWNLOAD_STATUS 1 HYFLUID_DOWNLOAD_MESSAGE)
    if (NOT HYFLUID_DOWNLOAD_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h: [${HYFLUID_DOWNLOAD_CODE}] ${HYFLUID_DOWNLOAD_MESSAGE}")
    endif ()
endif ()

set(HYFLUID_STB_IMAGE_WRITE_HEADER "${HYFLUID_VENDOR_DIR}/stb/stb_image_write.h")
if (EXISTS "${HYFLUID_STB_IMAGE_WRITE_HEADER}")
    message(STATUS "Using cached file: ${HYFLUID_STB_IMAGE_WRITE_HEADER}")
else ()
    file(MAKE_DIRECTORY "${HYFLUID_VENDOR_DIR}/stb")
    message(STATUS "Downloading https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h")
    file(DOWNLOAD
            "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
            "${HYFLUID_STB_IMAGE_WRITE_HEADER}"
            TLS_VERIFY ON
            STATUS HYFLUID_DOWNLOAD_STATUS
    )
    list(GET HYFLUID_DOWNLOAD_STATUS 0 HYFLUID_DOWNLOAD_CODE)
    list(GET HYFLUID_DOWNLOAD_STATUS 1 HYFLUID_DOWNLOAD_MESSAGE)
    if (NOT HYFLUID_DOWNLOAD_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h: [${HYFLUID_DOWNLOAD_CODE}] ${HYFLUID_DOWNLOAD_MESSAGE}")
    endif ()
endif ()

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
target_include_directories(hyfluid PRIVATE "${HYFLUID_VENDOR_DIR}" "${HYFLUID_FFMPEG_INCLUDE_DIR}")
target_link_libraries(hyfluid PRIVATE CUDA::cudart ${HYFLUID_FFMPEG_LIBRARIES})
set_property(TARGET hyfluid PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Executable
# ============================================================================
add_executable(hyfluid-app main.cpp)
target_link_libraries(hyfluid-app PRIVATE hyfluid)
if (WIN32)
    add_custom_command(
            TARGET hyfluid-app POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${HYFLUID_FFMPEG_RUNTIME_FILES} "$<TARGET_FILE_DIR:hyfluid-app>"
    )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET hyfluid-app PROPERTY BUILD_RPATH "${HYFLUID_FFMPEG_LIB_DIR}")
endif ()
set_property(TARGET hyfluid-app PROPERTY CXX_MODULE_STD ON)
