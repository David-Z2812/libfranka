// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <cmath>
#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "examples_common.h"
#include <franka/cartesian_impedance_controller.h>

/**
 * @example cartesian_impedance_control.cpp
 * An example showing a simple cartesian impedance controller without inertia shaping
 * that renders a spring damper system where the equilibrium is the initial configuration.
 * After starting the controller try to push the robot around and try different stiffness levels.
 *
 * @warning collision thresholds are set to high values. Make sure you have the user stop at hand!
 */

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  try {
    // connect to robot
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);
    // load the kinematics and dynamics model
    franka::Model model = robot.loadModel();

    std::cout << "This example will run the cartesian impedance controller and move the "
              << "endeffector by +10cm in X over 8 seconds. Make sure the workspace is "
              << "clear and have the user stop at hand. Press Enter to continue..." << std::endl;
    std::cin.ignore();

    // Use the helper to move relative: 0.10 m along X for 8 seconds
    franka_examples::move_relative(robot, model, Eigen::Vector3d(0.10, 0.0, 0.0), 8.0);

  } catch (const franka::Exception& ex) {
    std::cout << ex.what() << std::endl;
    return -1;
  }

  return 0;
}
