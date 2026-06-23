# [NeurIPS 2023] Inferring Hybrid Neural Fluid Fields from Videos. Hong-Xing Yu, et al.

[![Arch Linux](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/arch-build.yml)
[![Windows](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/windows-build.yml)

## 1. Algorithm Pipeline

On dev.

## 2. Build Instructions

#### Build Core Library

- CMake 4.3.0 or higher
- Ninja build system (for CXX std module support)
- A C++23 compliant compiler (tested on Arch Linux with gcc/g++ 15.2.1, Windows with MSVC 17.14.29)
- NVIDIA CUDA 13.2 or higher

```
cmake -B build -S . -G Ninja
cmake --build build --parallel 30
```

The default build only produces the core `libhyfluid` library. Dataset loading, CLI parsing, and runnable/plugin entrypoints stay outside the core library. App and plugin builds download pinned dataset loaders from `Xayah-Graphics/dataset`; app builds also download the pinned CLI module from `Xayah-Graphics/util`.

#### Build App

```
cmake -B build-app -S . -G Ninja -DHYFLUID_BUILD_APP=ON -DHYFLUID_TRAIN_PROFILE=baseline
cmake --build build-app --parallel 30
build-app\hyfluid.exe --help
```

#### Build Spectra Project Plugin

```
cmake -B build-plugin -S . -G Ninja -DHYFLUID_BUILD_PROJECT_PLUGIN=ON -DHYFLUID_TRAIN_PROFILE=baseline
cmake --build build-plugin --parallel 30
```

#### Build App And Plugin

```
cmake -B build-full -S . -G Ninja -DHYFLUID_BUILD_APP=ON -DHYFLUID_BUILD_PROJECT_PLUGIN=ON -DHYFLUID_TRAIN_PROFILE=baseline
cmake --build build-full --parallel 30
```
