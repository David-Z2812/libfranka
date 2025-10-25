// Example: Cartesian impedance controller driven by a TorchScript policy
// Loads a trained PyTorch model (TorchScript .pt) and, at a slow rate, feeds
// observations built from robot state to the network. The first 3 action
// components are interpreted as a relative translation command and executed
// via franka_examples::move_relative.

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>

#include <Eigen/Dense>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "examples_common.h"
#include <franka/cartesian_impedance_controller.h>

// libtorch
#include <torch/script.h>

namespace {

// Build a simple observation vector roughly following the provided obs_order:
// [fingertip_pos_rel_fixed(3), fingertip_quat(4), ee_linvel(3), ee_angvel(3),
//  ft_force(3), force_threshold(1), prev_actions(7)]
// Assumptions:
// - "fixed" frame is the robot's initial EE pose at program start
// - velocities are approximated by finite differences w.r.t. previous step
// - force is taken from RobotState.O_F_ext_hat_K first three entries (N)
struct ObsBuilder {
  Eigen::Vector3d fixed_pos;           // initial EE position
  Eigen::Quaterniond fixed_quat;       // initial EE orientation
  Eigen::Vector3d last_pos;            // for finite-difference linear velocity
  Eigen::Quaterniond last_quat;        // for finite-difference angular velocity
  std::array<double, 7> prev_actions{};  // last action sent (7-dim)
  bool initialized{false};

  void init_from_state(const franka::RobotState& s) {
    Eigen::Affine3d T(Eigen::Matrix4d::Map(s.O_T_EE.data()));
    fixed_pos = T.translation();
    fixed_quat = Eigen::Quaterniond(T.rotation());
    last_pos = fixed_pos;
    last_quat = fixed_quat;
    initialized = true;
  }

  torch::Tensor build(const franka::RobotState& s, double dt, double force_threshold) {
    if (!initialized) {
      init_from_state(s);
    }

    Eigen::Affine3d T(Eigen::Matrix4d::Map(s.O_T_EE.data()));
    Eigen::Vector3d pos = T.translation();
    Eigen::Quaterniond quat(T.rotation());

    // fingertip_pos_rel_fixed
    Eigen::Vector3d pos_rel = pos - fixed_pos;

    // velocities (simple finite difference)
    Eigen::Vector3d lin_vel = Eigen::Vector3d::Zero();
    if (dt > 0.0) {
      lin_vel = (pos - last_pos) / dt;
    }
    // approximate angular velocity from quaternion delta
    Eigen::Quaterniond dq = last_quat.conjugate() * quat;  // rotation from last to current
    if (dq.w() < 0.0) dq.coeffs() = -dq.coeffs();
    Eigen::AngleAxisd aa(dq);
    Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
    if (dt > 0.0) {
      ang_vel = (aa.axis() * aa.angle()) / dt;
    }

    // external force estimate in base frame
    Eigen::Vector3d ft_force(s.O_F_ext_hat_K[0], s.O_F_ext_hat_K[1], s.O_F_ext_hat_K[2]);

    // pack into tensor (float)
    std::vector<float> obs;
    obs.reserve(24);
    // pos_rel (3)
    for (int i = 0; i < 3; ++i) obs.push_back(static_cast<float>(pos_rel[i]));
    // quat (x,y,z,w) -> using Eigen order (x,y,z,w)
    obs.push_back(static_cast<float>(quat.x()));
    obs.push_back(static_cast<float>(quat.y()));
    obs.push_back(static_cast<float>(quat.z()));
    obs.push_back(static_cast<float>(quat.w()));
    // lin vel (3)
    for (int i = 0; i < 3; ++i) obs.push_back(static_cast<float>(lin_vel[i]));
    // ang vel (3)
    for (int i = 0; i < 3; ++i) obs.push_back(static_cast<float>(ang_vel[i]));
    // ft_force (3)
    for (int i = 0; i < 3; ++i) obs.push_back(static_cast<float>(ft_force[i]));
    // force_threshold (1)
    obs.push_back(static_cast<float>(force_threshold));
    // prev_actions (7)
    for (double v : prev_actions) obs.push_back(static_cast<float>(v));

    // update last pose
    last_pos = pos;
    last_quat = quat;

    // create 1xN tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t = torch::from_blob(obs.data(), {(long)1, (long)obs.size()}, options).clone();
    return t;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname> <policy.pt> [duration_s]" << std::endl;
    return -1;
  }

  const char* hostname = argv[1];
  const std::string model_path = argv[2];
  const double total_duration_s = (argc >= 4) ? std::atof(argv[3]) : 15.0;

  try {
    franka::Robot robot(hostname);
    setDefaultBehavior(robot);
    franka::Model model = robot.loadModel();

    // Load TorchScript policy
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();
    torch::NoGradGuard no_grad;

    // Prepare observation builder
    ObsBuilder ob;

    // Some parameters
    const double step_s = 0.2;         // execute each NN action over this duration
    const double pos_scale = 0.02;     // scale factor for position actions (meters per unit)
    const double force_threshold = 10.0;  // N, placeholder

    std::cout << "WARNING: This example will move the robot using a NN policy. "
              << "Ensure a safe workspace and have the user stop ready. Press Enter to continue..." << std::endl;
    std::cin.ignore();

    auto start_tp = std::chrono::steady_clock::now();
    auto prev_tp = start_tp;

    while (true) {
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start_tp).count();
      if (elapsed >= total_duration_s) {
        std::cout << "Finished NN-controlled session." << std::endl;
        break;
      }
      double dt = std::chrono::duration<double>(now - prev_tp).count();
      prev_tp = now;

      // Read current state
      franka::RobotState state = robot.readOnce();

      // Build observation tensor
      torch::Tensor obs = ob.build(state, dt, force_threshold);

      // Forward through policy
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(obs);
      torch::Tensor action = module.forward(inputs).toTensor();
      if (action.dim() == 2 && action.size(0) == 1) {
        action = action.squeeze(0);
      }
      if (action.size(0) < 7) {
        throw std::runtime_error("Policy output must have at least 7 elements (got " + std::to_string(action.size(0)) + ")");
      }

      // Extract translation command (first 3)
      Eigen::Vector3d delta(action[0].item<float>(), action[1].item<float>(), action[2].item<float>());
      // scale and clip a bit for safety
      delta = pos_scale * delta;
      for (int i = 0; i < 3; ++i) {
        if (delta[i] > 0.05) delta[i] = 0.05;   // max 5 cm per step
        if (delta[i] < -0.05) delta[i] = -0.05;
      }

      // store prev actions
      for (int i = 0; i < 7; ++i) {
        ob.prev_actions[i] = (i < action.size(0)) ? action[i].item<float>() : 0.0;
      }

      // Execute via impedance controller helper
      franka_examples::move_relative(robot, model, delta, step_s);
    }

  } catch (const c10::Error& e) {
    std::cerr << "Torch error: " << e.msg() << std::endl;
    return -1;
  } catch (const franka::Exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
