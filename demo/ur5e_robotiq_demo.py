"""
Demo script for UR5e with Robotiq 2F-85 gripper
Shows how to use the UR5e arm with the Robotiq gripper attached
"""
import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, Robotiq2F85
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController

# Create the arena
arena = StandardArena()

# Create mocap target for OSC
target = Target(arena.mjcf_model)

# Create UR5e arm
ur5e = Arm(
    xml_path=os.path.join(
        os.path.dirname(__file__),
        '../manipulator_mujoco/assets/robots/ur5e/ur5e.xml',
    ),
    eef_site_name='eef_site',
    attachment_site_name='attachment_site',
    joint_names=[
        'shoulder_pan_joint',
        'shoulder_lift_joint', 
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
)

# Create Robotiq 2F-85 gripper
gripper = Robotiq2F85()

# Attach gripper to arm
ur5e.attach_tool(gripper.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])

# Attach arm to arena
arena.attach(ur5e.mjcf_model, pos=[0, 0, 0])

# Create physics
physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

# Set up OSC controller for the arm
controller = OperationalSpaceController(
    physics=physics,
    joints=ur5e.joints,
    eef_site=ur5e.eef_site,
    min_effort=-150.0,
    max_effort=150.0,
    kp=300,
    ko=300,
    kv=30,
    vmax_xyz=1.0,
    vmax_abg=2.0,
)

# Initialize the simulation
with physics.reset_context():
    # Set arm to home position
    physics.bind(ur5e.joints).qpos = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    # Set target position
    target.set_mocap_pose(physics, position=[0.5, 0, 0.3], quaternion=[0, 0, 0, 1])
    # Initialize gripper (closed position)
    physics.bind(gripper.actuator).ctrl = 0

# Launch the viewer
viewer = mujoco.viewer.launch_passive(
    physics.model.ptr,
    physics.data.ptr,
)

timestep = physics.model.opt.timestep
step_start = time.time()

print("UR5e with Robotiq 2F-85 Gripper Demo")
print("=====================================")
print("The arm will move to follow the mocap target.")
print("The gripper will open and close periodically.")
print("Press Ctrl+C to exit.")

gripper_state = 0  # 0 = closed, 255 = open
gripper_direction = 1
step_count = 0

try:
    while viewer.is_running():
        # Get target pose from mocap
        target_pose = target.get_mocap_pose(physics)
        
        # Run OSC controller
        controller.run(target_pose)
        
        # Control gripper - open and close periodically
        if step_count % 500 == 0:
            gripper_direction *= -1
        
        if gripper_direction > 0:
            gripper_state = min(255, gripper_state + 2)
        else:
            gripper_state = max(0, gripper_state - 2)
        
        physics.bind(gripper.actuator).ctrl = gripper_state
        
        # Step physics
        physics.step()
        
        # Update viewer
        viewer.sync()
        
        # Frame rate control
        time_until_next_step = timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        step_start = time.time()
        step_count += 1

except KeyboardInterrupt:
    print("\nDemo stopped by user.")

# Close the viewer
viewer.close()
print("Demo finished.")
