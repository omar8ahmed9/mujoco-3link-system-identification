# ROB701 — Identification of a 3-Link Planar Manipulator

## 🧭 Overview
This project identifies the dynamic parameters of a **3-link planar manipulator** modeled in **MuJoCo**.
The goal is to reproduce the full process described in Chapter 7 of *Robotics: Modelling, Planning and Control* — from model setup to validation.

---

## 🧩 Contents

| File | Description |
|------|--------------|
| `three_link_pendulum_vertical.xml` | MuJoCo model of the 3-link planar manipulator (modified parameters). |
| `trajectory_test.py` | Generates and visualizes a persistently-exciting (multi-sine) trajectory. |
| `log_run.py` | Runs the simulation with a PD controller and logs joint states and torques to `run_log.csv`. |
| `run_log.csv` | Recorded data: time, joint positions/velocities, torques, and tip position. |
| `estimate_params.py` | Builds the regressor and estimates dynamic parameters using least-squares. |
| `identified_params.csv` | Estimated parameters (inertias, masses, friction). |
| `plot_check.py` | Compares measured vs predicted torques and prints conditioning metrics. |
| `torque_plot.png` | Overlay of measured and predicted torques for validation. |
| `report_3R_ID.tex` | LaTeX report template summarizing objectives, method, and results. |
| `README.md` | This file — overview and usage instructions. |

---

## ⚙️ Requirements

- Python ≥ 3.10  
- MuJoCo ≥ 3.3  
- Packages:
  ```bash
  pip install mujoco mujoco-python-viewer numpy matplotlib


## 🚀 How to Run

1. Open and Verify the Model
    ```bash
    python3 -m mujoco.viewer three_link_pendulum_vertical.xml
Ensure the arm hangs vertically and moves in the x–z plane.

2. Generate the Trajectory
   ```bash
   python3 trajectory_test.py
Runs the multi-sine motion to verify joint limits and smoothness.

4. Estimate Parameters
   ```bash
    python3 estimate_params.py
Outputs: identified_params.csv
Prints parameter estimates and joint RMSE.

5. Validate Results
   ```bash
   python3 plot_check.py
Shows torque comparison plots and prints regressor conditioning.

## 📊 Results Summary

RMSE per joint: [0.019, 0.014, 0.009] Nm

Condition number: 33.5 (well-conditioned trajectory)

Identified parameters match MJCF truth within 0.1–0.3%.

Measured vs predicted torques overlap perfectly.


