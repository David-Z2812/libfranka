// Implementation of cartesian impedance helper
// Uses a torque-level Cartesian impedance controller and a Cartesian pose
// callback that moves the equilibrium by a relative translation over a
// specified duration.
// Copyright (c) 2025

#include <franka/cartesian_impedance_controller.h>

#include <array>
#include <cmath>
#include <functional>
#include <iostream>

#include <franka/duration.h>
#include <franka/exception.h>

namespace franka_examples {

void move_relative(franka::Robot& robot, franka::Model& model,
                   const Eigen::Vector3d& translation, double duration_sec) {
  if (duration_sec <= 0.0) {
    throw std::invalid_argument("duration_sec must be positive");
  }

  // Compliance parameters (same as original example)
  const double translational_stiffness{150.0};
  const double rotational_stiffness{10.0};
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                     Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                         Eigen::MatrixXd::Identity(3, 3);

  // set collision behavior (use same permissive values as original example)
  robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                             {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                             {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                             {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

  // desired pose variables shared between callbacks
  Eigen::Vector3d position_d;
  Eigen::Quaterniond orientation_d;

  // Read initial state to initialize desired pose
  franka::RobotState initial_state = robot.readOnce();
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
  position_d = Eigen::Vector3d(initial_transform.translation());
  orientation_d = Eigen::Quaterniond(initial_transform.rotation());

  // Torque callback: Cartesian impedance controller using the shared desired pose
  std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
      impedance_control_callback =
          [&](const franka::RobotState& robot_state, franka::Duration /*period*/) -> franka::Torques {
    // get state variables
    std::array<double, 7> coriolis_array = model.coriolis(robot_state);
    std::array<double, 42> jacobian_array =
        model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

    // convert to Eigen
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
    Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d position(transform.translation());
    Eigen::Quaterniond orientation(transform.rotation());

    // compute error to desired equilibrium pose
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d;

    // orientation error
    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
      orientation.coeffs() << -orientation.coeffs();
    }
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    // Transform to base frame
    error.tail(3) << -transform.rotation() * error.tail(3);

    // compute control
    Eigen::VectorXd tau_task(7), tau_d(7);
    tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
    tau_d << tau_task + coriolis;

    std::array<double, 7> tau_d_array{};
    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
    return tau_d_array;
  };

  // Cartesian pose callback: updates desired pose over time and finishes after duration
  std::array<double, 16> initial_pose_array;
  double time = 0.0;
  auto cartesian_pose_callback = [&](const franka::RobotState& robot_state,
                                     franka::Duration period) -> franka::CartesianPose {
    time += period.toSec();
    if (time == 0.0) {
      initial_pose_array = robot_state.O_T_EE;
    }

    // smooth interpolation (cosine) from 0 to 1 over duration
    double s = 1.0;
    if (time < duration_sec) {
      s = 0.5 * (1.0 - std::cos(M_PI * time / duration_sec));
    }

    std::array<double, 16> new_pose = initial_pose_array;
    new_pose[12] += s * translation.x();
    new_pose[13] += s * translation.y();
    new_pose[14] += s * translation.z();

    // update the shared desired pose used by impedance controller
    position_d << new_pose[12], new_pose[13], new_pose[14];
    // keep orientation fixed as initial orientation
    // Angular part remains the same as initial transform
    Eigen::Affine3d current_initial(Eigen::Matrix4d::Map(initial_pose_array.data()));
    orientation_d = Eigen::Quaterniond(current_initial.rotation());

    if (time >= duration_sec) {
      std::cout << std::endl << "Finished move_relative motion." << std::endl;
      return franka::MotionFinished(new_pose);
    }
    return new_pose;
  };

  // Start real-time control loop with both callbacks. The cartesian pose callback
  // will finish the motion after duration_sec.
  robot.control(impedance_control_callback, cartesian_pose_callback);
}

}  // namespace franka_examples
