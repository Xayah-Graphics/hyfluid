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
    target_link_libraries(hyfluid PRIVATE xayah-util-xcli libhyfluid)
    set_property(TARGET hyfluid PROPERTY CXX_MODULE_STD ON)
endif ()
