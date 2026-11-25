cmake_minimum_required(VERSION 3.16)
project(auralis_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(auralis SHARED src/aur_native.cpp src/aur_grid.cpp)
# the grid builder must reproduce the Python reference bit for bit: no FMA contraction
if(MSVC)
  set_source_files_properties(src/aur_grid.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
else()
  set_source_files_properties(src/aur_grid.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

if(MSVC)
  target_compile_options(auralis PRIVATE /O2 /EHsc)
else()
  target_compile_options(auralis PRIVATE -O3)
  find_package(Threads REQUIRED)
  target_link_libraries(auralis PRIVATE Threads::Threads)
endif()

# Drop the library exactly where the Python package looks for it:
#   auralis/lib/<os>-<arch>/
if(WIN32)
  set(AUR_OS "windows")
elseif(APPLE)
  set(AUR_OS "macos")
else()
  set(AUR_OS "linux")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" AUR_ARCH)
if(AUR_ARCH MATCHES "amd64|x86_64")
  set(AUR_ARCH "x86_64")
elseif(AUR_ARCH MATCHES "arm64|aarch64")
  # the same platform tag as auralis/engine/native.py: linux-aarch64, macos-arm64
  if(APPLE)
    set(AUR_ARCH "arm64")
  else()
    set(AUR_ARCH "aarch64")
  endif()
endif()

set(AUR_OUT "${CMAKE_CURRENT_SOURCE_DIR}/../auralis/lib/${AUR_OS}-${AUR_ARCH}")
set_target_properties(auralis PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${AUR_OUT}"
  RUNTIME_OUTPUT_DIRECTORY "${AUR_OUT}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${AUR_OUT}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${AUR_OUT}"
  PREFIX "lib"
  OUTPUT_NAME "auralis")

message(STATUS "Auralis: the library will be written to ${AUR_OUT}")
