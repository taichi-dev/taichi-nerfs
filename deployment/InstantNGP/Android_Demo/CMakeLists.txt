cmake_minimum_required(VERSION 3.13)

project(TaichiAotDemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

####################################################
# Configure Environment: Taichi C-API Installation #
####################################################
set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})
find_library(taichi_c_api taichi_c_api HINTS
    ${TAICHI_C_API_INSTALL_DIR}/lib
    NO_CMAKE_FIND_ROOT_PATH)

if (NOT EXISTS ${taichi_c_api})
    message(FATAL_ERROR "Couldn't find C-API library; ensure your Taichi is built with `TI_WITH_CAPI=ON`")
endif()

##########################
# Build for Android Demo #
##########################
set(TAICHI_NERF_DEMO_NAME "nerf")
set(PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")

list(APPEND TAICHI_TUTORIAL_DEMO_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${PROJECT_ROOT_DIR}/utils/utils.cpp
  ${PROJECT_ROOT_DIR}/utils/app_fp32.cpp
)

#execute_process(COMMAND python3 "${PROJECT_ROOT_DIR}/taichi_ngp/taichi_ngp.py" --scene smh_lego --aot --res_w=300 --res_h=600)

add_executable(${TAICHI_NERF_DEMO_NAME} ${TAICHI_TUTORIAL_DEMO_SRC})
target_include_directories(${TAICHI_NERF_DEMO_NAME} PUBLIC
  ${PROJECT_ROOT_DIR}/utils
  ${TAICHI_C_API_INSTALL_DIR}/include)

target_link_libraries(${TAICHI_NERF_DEMO_NAME} ${taichi_c_api})

set_target_properties(${TAICHI_NERF_DEMO_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

if (ANDROID)
  target_link_libraries(${TAICHI_NERF_DEMO_NAME} android log)
endif()
