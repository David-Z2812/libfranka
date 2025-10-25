cmake_minimum_required(VERSION 3.4)

project(libfranka-examples CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/../cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Load dependencies
include(FetchFMT)

if(NOT FRANKA_IS_FOUND)
    find_package(Franka REQUIRED)
endif()
find_package(Eigen3 REQUIRED)
find_package(Poco REQUIRED COMPONENTS Foundation)
find_package(pinocchio REQUIRED)
find_package(Torch QUIET)
if(Torch_FOUND)
  # Some Torch packages export Torch::Torch, others only variables.
  if(TARGET Torch::Torch)
    set(TORCH_TARGET Torch::Torch)
  else()
    # Fallback to variable-based linking
    set(TORCH_TARGET ${TORCH_LIBRARIES})
  endif()
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(examples_common STATIC
  examples_common.cpp
  cartesian_impedance_controller.cpp
)

target_link_libraries(examples_common PUBLIC Franka::Franka Eigen3::Eigen3)
target_include_directories(examples_common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set(EXAMPLES
  cartesian_impedance_control
  cartesian_impedance_control_example
  communication_test
  echo_robot_state
  force_control
  generate_cartesian_pose_motion
  generate_cartesian_pose_motion_external_control_loop
  generate_cartesian_velocity_motion
  generate_cartesian_velocity_motion_external_control_loop
  generate_consecutive_motions
  generate_elbow_motion
  generate_joint_position_motion
  generate_joint_position_motion_external_control_loop
  generate_joint_velocity_motion
  generate_joint_velocity_motion_external_control_loop
  grasp_object
  joint_impedance_control
  joint_point_to_point_motion
  motion_with_control
  motion_with_control_external_control_loop
  print_joint_poses
  vacuum_object
)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_include_directories(${example} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${example} Franka::Franka examples_common Eigen3::Eigen3 pinocchio::pinocchio)
endforeach()

target_link_libraries(joint_impedance_control Threads::Threads pinocchio::pinocchio)
target_link_libraries(motion_with_control Poco::Foundation)
target_link_libraries(motion_with_control_external_control_loop Poco::Foundation)

# TorchScript-based NN controller example (optional)
if(Torch_FOUND)
  message(STATUS "Found libtorch: enabling cartesian_impedance_control_nn example")
  add_executable(cartesian_impedance_control_nn cartesian_impedance_control_nn.cpp)
  target_link_libraries(cartesian_impedance_control_nn PRIVATE
    Franka::Franka examples_common Eigen3::Eigen3 pinocchio::pinocchio ${TORCH_TARGET})
  if (TORCH_INCLUDE_DIRS)
    target_include_directories(cartesian_impedance_control_nn PRIVATE ${TORCH_INCLUDE_DIRS})
  endif()
  # Torch can require special C++ flags
  if (TORCH_CXX_FLAGS)
    target_compile_options(cartesian_impedance_control_nn PRIVATE ${TORCH_CXX_FLAGS})
  endif()
  install(TARGETS cartesian_impedance_control_nn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
  message(STATUS "libtorch not found - skipping cartesian_impedance_control_nn example")
endif()

include(GNUInstallDirs)
install(TARGETS ${EXAMPLES}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Further examples
add_subdirectory(utility_examples)
