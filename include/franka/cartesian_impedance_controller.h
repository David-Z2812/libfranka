// Header for cartesian impedance helper
// Provides move_relative to run a cartesian impedance controller that moves the
// equilibrium pose relative to the current end effector pose over a duration.
// Copyright (c) 2025
#pragma once

#include <Eigen/Dense>
#include <franka/model.h>
#include <franka/robot.h>

namespace franka_examples {

// Move the robot's end effector by `translation` (in meters) relative to its
// current pose using a Cartesian impedance controller. The motion will be
// executed over approximately `duration_sec` seconds and then the controller
// will finish. This function blocks until the motion finished or an exception
// is thrown.
// Parameters:
// - robot: connected franka::Robot instance (will be used and not closed)
// - model: kinematics/dynamics model from robot.loadModel()
// - translation: desired relative translation in base frame (x,y,z)
// - duration_sec: motion duration in seconds (> 0)
void move_relative(franka::Robot& robot, franka::Model& model,
                   const Eigen::Vector3d& translation, double duration_sec);

}  // namespace franka_examples
