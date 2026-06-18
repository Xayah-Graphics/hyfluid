# [NeurIPS 2023] Inferring Hybrid Neural Fluid Fields from Videos. Hong-Xing Yu, et al.

[![Linux Build (Arch)](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/arch-build.yml)
[![Windows Build](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/hyfluid/actions/workflows/windows-build.yml)

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

The default build only produces the core `libhyfluid` library. Dataset loading, CLI parsing, and the runnable app are intentionally kept outside the core library. App builds download pinned dataset loaders from `Xayah-Graphics/dataset` and the pinned CLI module from `Xayah-Graphics/util`.

#### Build App

```
cmake -B build-app -S . -G Ninja -DHYFLUID_BUILD_APP=ON -DHYFLUID_TRAIN_PROFILE=baseline
cmake --build build-app --parallel 30
build-app\hyfluid.exe --help
```
