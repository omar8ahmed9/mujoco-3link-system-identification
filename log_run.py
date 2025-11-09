# log_run.py
import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

XML = "three_link_pendulum_vertical.xml"

# --- timing & trajectory ---
T_total = 12.0          # seconds (a little longer than before)
dt = 0.002              # simulation/control step
t = np.arange(0, T_total, dt)
N = t.size

# multi-sine trajectory (same design as before)
A = np.array([[0.6,0.3,0.2],
              [0.5,0.25,0.15],
              [0.4,0.2,0.1]])
F = np.array([[0.2,0.5,0.9],
              [0.3,0.7,1.1],
              [0.4,0.8,1.3]])
rng = np.random.default_rng(0)
PH = rng.uniform(0, 2*np.pi, (3,3))

q_ref  = np.zeros((N,3))
qd_ref = np.zeros_like(q_ref)
qdd_ref= np.zeros_like(q_ref)
for i in range(3):
    for k in range(3):
        q_ref[:,i]  += A[i,k]*np.sin(2*np.pi*F[i,k]*t + PH[i,k])
        qd_ref[:,i] += A[i,k]*(2*np.pi*F[i,k])*np.cos(2*np.pi*F[i,k]*t + PH[i,k])
        qdd_ref[:,i]+= -A[i,k]*(2*np.pi*F[i,k])**2*np.sin(2*np.pi*F[i,k]*t + PH[i,k])

# --- PD tracking gains (conservative) ---
KP = np.array([8.0, 8.0, 8.0])
KD = np.array([0.8, 0.8, 0.8])

# --- MuJoCo setup ---
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)
model.opt.timestep = dt  # keep sim step in sync with our dt

# CSV writer
out = Path("run_log.csv").resolve()
header = [
    "t",
    "q1","q2","q3",
    "qd1","qd2","qd3",
    "tau1","tau2","tau3",
    "tip_x","tip_y","tip_z"
]
f = open(out, "w", newline="")
writer = csv.writer(f)
writer.writerow(header)

# Helper to read actuator torques safely:
def read_torque(data):
    # If you added <sensor><actuatorfrc actuator="m1/m2/m3">...</sensor>,
    # you can also pull from data.sensordata. Using data.actuator_force is simpler here.
    return np.copy(data.actuator_force)

# Run with viewer and log
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Logging to:", out)
    for k in range(N):
        # PD control to track q_ref
        e  = q_ref[k]  - data.qpos
        ed = qd_ref[k] - data.qvel
        data.ctrl[:] = KP*e + KD*ed

        mujoco.mj_step(model, data)

        # Read sensors/state
        tau = read_torque(data)

        # tip position (world frame)
        # robust name->id lookup across MuJoCo Python versions
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        except Exception:
            # fallback: no name lookup available -> assume site 0 (or skip logging)
            site_id = 0
        tip = data.site_xpos[site_id]


        writer.writerow([
            t[k],
            data.qpos[0], data.qpos[1], data.qpos[2],
            data.qvel[0], data.qvel[1], data.qvel[2],
            tau[0], tau[1], tau[2],
            tip[0], tip[1], tip[2],
        ])

        # show
        viewer.sync()
        # optional sleep to match real-time; not required for correct logging
        time.sleep(dt)

f.close()
print("Done. Saved:", out)
