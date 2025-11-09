import numpy as np
import mujoco
import mujoco.viewer
import time

# Load your model
model = mujoco.MjModel.from_xml_path("three_link_pendulum_vertical.xml")
data  = mujoco.MjData(model)

# time vector
T_total = 10.0
dt = 0.002
t = np.arange(0, T_total, dt)

# multi-sine coefficients
A = np.array([[0.6,0.3,0.2],
              [0.5,0.25,0.15],
              [0.4,0.2,0.1]])
F = np.array([[0.2,0.5,0.9],
              [0.3,0.7,1.1],
              [0.4,0.8,1.3]])
rng = np.random.default_rng(0)
PH = rng.uniform(0,2*np.pi,(3,3))

# precompute q, qd, qdd
N = t.size
q = np.zeros((N,3)); qd = np.zeros_like(q); qdd = np.zeros_like(q)
for i in range(3):
    for k in range(3):
        q[:,i]  += A[i,k]*np.sin(2*np.pi*F[i,k]*t + PH[i,k])
        qd[:,i] += A[i,k]*(2*np.pi*F[i,k])*np.cos(2*np.pi*F[i,k]*t + PH[i,k])
        qdd[:,i]+= -A[i,k]*(2*np.pi*F[i,k])**2*np.sin(2*np.pi*F[i,k]*t + PH[i,k])

# run in viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    for k in range(N):
        data.ctrl[:] = 5.0*(q[k] - data.qpos) - 0.5*data.qvel  # simple PD to track q
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
