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
