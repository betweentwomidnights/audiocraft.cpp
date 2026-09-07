cmake_minimum_required(VERSION 3.20)
project(audiocraft_cpp LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put our binaries next to the ggml DLLs so tools run without PATH juggling.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Backend toggles (forwarded to the ggml submodule).
option(AC_CUDA   "Build the GGML CUDA backend"     OFF)
option(AC_VULKAN "Build the GGML Vulkan backend"   OFF)
option(AC_METAL  "Build the GGML Metal backend"    OFF)
option(AC_HIP    "Build the GGML HIP/ROCm backend" OFF)

option(AC_BUILD_TOOLS "Build the CLI tools and tests" ON)

# Model families. Each is one STATIC library from one .cpp plus headers, mirroring how
# sa3.cpp gates its stable-audio-tools family behind SA3_BUILD_SAT. Kept independent of
# AC_BUILD_TOOLS so an embedding application can link a family without building any CLI.
# Both default OFF until their phase lands; see docs/ROADMAP.md.
option(AC_BUILD_MELODYFLOW "Build the MelodyFlow family (DiT + VAE + flow solver)" OFF)
option(AC_BUILD_MUSICGEN   "Build the MusicGen family (LM + EnCodec 32kHz)"        OFF)

if(AC_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # CUDA graphs need Ampere+ (CC >= 8.0). Only force them off when we can PROVE the build
    # targets exclusively pre-Ampere GPUs: every CMAKE_CUDA_ARCHITECTURES entry is a plain
    # number < 80. Any non-numeric token (native/all/all-major) means unknown at configure
    # time, so we leave the ggml default alone.
    if(CMAKE_CUDA_ARCHITECTURES)
        set(_ac_all_pre_ampere ON)
        set(_ac_any_numeric OFF)
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            string(REGEX REPLACE "-(real|virtual)$" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                set(_ac_any_numeric ON)
                if(_arch_num GREATER_EQUAL 80)
                    set(_ac_all_pre_ampere OFF)
                    break()
                endif()
            else()
                set(_ac_all_pre_ampere OFF)
                break()
            endif()
        endforeach()
        if(_ac_any_numeric AND _ac_all_pre_ampere)
            set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
            message(STATUS "AC_CUDA: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} is all pre-Ampere; forcing GGML_CUDA_GRAPHS=OFF")
        endif()
    endif()
endif()
if(AC_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(AC_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(AC_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

# audiocraft.cpp pins an exact revision of the shared betweentwomidnights/ggml fork -- the same
# pin sa3.cpp uses, so the two repos stay trivially comparable. See docs/GGML_FORK.md.
# Fresh checkouts not cloned with --recurse-submodules have an empty ggml/; initialize it here so
# clean configure runs do not depend on out-of-tree copies.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        message(STATUS "Initializing ggml submodule")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive ggml
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
        message(FATAL_ERROR
            "ggml checkout is missing. Run `git submodule update --init --recursive ggml` "
            "from ${CMAKE_CURRENT_SOURCE_DIR} and reconfigure.")
    endif()
endif()

# Existing clones can retain an older submodule checkout after a parent-repository update.
# Detect it without modifying the checkout: a maintainer may have deliberate local ggml work.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD:ggml
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE AC_GGML_PIN_RESULT OUTPUT_VARIABLE AC_GGML_PIN
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C ggml rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE AC_GGML_HEAD_RESULT OUTPUT_VARIABLE AC_GGML_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(AC_GGML_PIN_RESULT EQUAL 0 AND AC_GGML_HEAD_RESULT EQUAL 0 AND
       NOT AC_GGML_PIN STREQUAL AC_GGML_HEAD)
        message(WARNING
            "ggml is checked out at ${AC_GGML_HEAD}, but this audiocraft.cpp revision pins "
            "${AC_GGML_PIN}. Run `git submodule sync --recursive` then "
            "`git submodule update --init --recursive`. Local ggml work was not modified.")
    endif()
endif()
add_subdirectory(ggml)

# src/ is header-driven; families add their own STATIC libraries below.
add_library(ac INTERFACE)
target_include_directories(ac INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ac INTERFACE ggml)

if(AC_BUILD_MELODYFLOW)
    add_library(ac_melodyflow STATIC src/mf/pipeline.cpp)
    target_link_libraries(ac_melodyflow PUBLIC ac)
endif()

if(AC_BUILD_MUSICGEN)
    add_library(ac_musicgen STATIC src/mg/pipeline.cpp)
    target_link_libraries(ac_musicgen PUBLIC ac)
endif()

# --- tools -----------------------------------------------------------------
if(AC_BUILD_TOOLS)
    add_executable(ac-textenc tools/ac-textenc.cpp)
    target_link_libraries(ac-textenc PRIVATE ac)

    add_executable(ac-tokenize tools/ac-tokenize.cpp)
    target_link_libraries(ac-tokenize PRIVATE ac)

    add_executable(ac-vae tools/ac-vae.cpp)
    target_link_libraries(ac-vae PRIVATE ac)
endif()

# --- tests -----------------------------------------------------------------
if(BUILD_TESTING AND AC_BUILD_TOOLS)
    # Prefer the repo venv: the converter tests need gguf/numpy/safetensors, which a bare
    # system interpreter usually lacks (they would silently skip instead of running).
    if(WIN32)
        set(_ac_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    else()
        set(_ac_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${_ac_venv_python}")
        set(AC_PYTHON "${_ac_venv_python}")
    else()
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            set(AC_PYTHON "${Python3_EXECUTABLE}")
        endif()
    endif()

    add_executable(shared_headers_test tests/shared_headers_test.cpp)
    target_link_libraries(shared_headers_test PRIVATE ac)
    add_test(NAME shared_headers COMMAND shared_headers_test)
    set_tests_properties(shared_headers PROPERTIES LABELS "non-gpu;shared")

    add_executable(lstm_test tests/lstm_test.cpp)
    target_link_libraries(lstm_test PRIVATE ac)
    add_test(NAME lstm COMMAND lstm_test)
    set_tests_properties(lstm PROPERTIES LABELS "non-gpu;shared")

    add_executable(seanet_shape_test tests/seanet_shape_test.cpp)
    target_link_libraries(seanet_shape_test PRIVATE ac)
    add_test(NAME seanet_shape COMMAND seanet_shape_test)
    set_tests_properties(seanet_shape PROPERTIES LABELS "non-gpu;melodyflow")

    add_executable(t5_bucket_test tests/t5_bucket_test.cpp)
    target_link_libraries(t5_bucket_test PRIVATE ac)
    add_test(NAME t5_bucket COMMAND t5_bucket_test)
    set_tests_properties(t5_bucket PROPERTIES LABELS "non-gpu;shared")

    if(AC_PYTHON)
        add_test(NAME t5_converter
                 COMMAND ${AC_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/t5_converter_test.py)
        set_tests_properties(t5_converter PROPERTIES LABELS "non-gpu;tooling")

        add_test(NAME seanet_converter
                 COMMAND ${AC_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/seanet_converter_test.py)
        set_tests_properties(seanet_converter PROPERTIES LABELS "non-gpu;tooling;melodyflow")
    endif()
endif()
