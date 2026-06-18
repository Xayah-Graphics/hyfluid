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

The default build only produces the core `hyfluid-train` library. Dataset loading, CLI parsing, and runnable training/evaluation entrypoints are intentionally kept outside the core library.

#### Build Benchmarks

Benchmark builds download dataset loader source files from `Xayah-Graphics/dataset` at the fixed commit configured by `HYFLUID_DATASET_GIT_TAG`. Only the `baseline` benchmark profile is built by default; later profiles can be enabled through `HYFLUID_BENCHMARK_PROFILES` without changing the benchmark structure.

```
cmake -B build-benchmarks -S . -G Ninja -DHYFLUID_BUILD_BENCHMARKS=ON
cmake --build build-benchmarks --parallel 30
build-benchmarks\hyfluid-benchmark-baseline.exe --help
```
