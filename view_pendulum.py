import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("three_link_pendulum_vertical.xml")
data = mujoco.MjData(model)

print("Press [space] pause/play, [r] reset, [Esc] quit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)  # advances physics
        viewer.sync()                 # updates the window
