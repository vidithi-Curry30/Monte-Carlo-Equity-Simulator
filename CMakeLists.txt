cmake_minimum_required(VERSION 3.16)
project(monte_carlo_simulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimisation: auto-vectorisation, native SIMD, fast math.
# -ffast-math lets the compiler reorder FP ops and use SIMD reductions.
add_compile_options(-O3 -march=native -ffast-math)

find_package(OpenMP)

# ── Main simulator binary ──────────────────────────────────────────────
add_executable(simulator src/main.cpp)
target_include_directories(simulator PRIVATE src)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simulator OpenMP::OpenMP_CXX)
endif()

# ── Benchmark binary ───────────────────────────────────────────────────
add_executable(bench benchmark/bench.cpp)
target_include_directories(bench PRIVATE src)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench OpenMP::OpenMP_CXX)
endif()
