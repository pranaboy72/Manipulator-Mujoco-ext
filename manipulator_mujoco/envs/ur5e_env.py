import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, Robotiq2F85, AG95
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.utils.transform_utils import mat2quat


def _as_vec3(v):
    return tuple(v) if v is not None else (0.0, 0.0, 0.0)

def _as_quat(v):
    return tuple(v) if v is not None else (1.0, 0.0, 0.0, 0.0)

def _pose_to_freejoint_qpos(pos, quat_xyzw):
    # freejoint expects [x y z qw qx qy qz]; we have [qx qy qz qw]
    qx, qy, qz, qw = quat_xyzw
    x, y, z = pos
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)

def _add_axes_geoms_on_body(body, prefix, length=0.08, radius=0.002):
    body.add('geom', name=f'{prefix}_dot', type='sphere', size=f'{radius*5:.3f}',
             rgba='1 1 1 1', contype='0', conaffinity='0', group='0')
    body.add('geom', name=f'{prefix}_x', type='capsule', fromto=f'0 0 0 {length} 0 0',
             size=f'{radius}', rgba='1 0 0 1', contype='0', conaffinity='0', group='0')
    body.add('geom', name=f'{prefix}_y', type='capsule', fromto=f'0 0 0 0 {length} 0',
             size=f'{radius}', rgba='0 1 0 1', contype='0', conaffinity='0', group='0')
    body.add('geom', name=f'{prefix}_z', type='capsule', fromto=f'0 0 0 0 0 {length}',
             size=f'{radius}', rgba='0 0 1 1', contype='0', conaffinity='0', group='0')


class UR5eEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, gripper=False, cube=False, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)

        # ur5e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        # robotiq 2f-85 gripper
        if gripper:
            self._gripper = Robotiq2F85()
            self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])
            
            # Adjust the EEF frame to be at the pinch site
            # pinch_site = self._gripper.mjcf_model.find("site", "pinch")
            # if pinch_site is not None:
            #     pinch_site.pos = [0, 0, 0.1]
            
            act = self._gripper.mjcf_model.find("actuator", "fingers_actuator")
            
            if act is not None:
                act.set_attributes(forcerange="-160 160")
                
        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
        
        # optional cube for pick-and-place tasks
        if cube:
            self._cube = Primitive(
                type='box',
                size=[0.02, 0.02, 0.02],  # 4cm cube (half-sizes)
                rgba=[1, 0, 0, 1],  # Red color
                mass=0.1,  # 100g
                name='cube',
                friction=[1, 0.3, 0.0001]
            )
            
            # Attach cube to arena with a free joint (so it can move)
            self._cube_frame = self._arena.attach_free(
                self._cube.mjcf_model,
                pos=[0.5, 0.134, 0.02]  # Start on table surface
            )
            
            self._arena.mjcf_model.option.cone = 'elliptic'
            
            if hasattr(self, '_gripper'):
                pad_geoms = self._gripper.mjcf_model.find_all('geom')
                for geom in pad_geoms:
                    if geom.name and 'pad' in geom.name and ('pad1' in geom.name or 'pad2' in geom.name):
                        geom.friction = [1.8, 0.8, 0.1] # High torsional friction

            for geom in self._cube.mjcf_model.find_all('geom'):
                geom.friction = [1.2, 0.5, 0.05]
                
        self._choose_eef_site(prefer_gripper_tcp=gripper, tcp_site_name='pinch', create_if_missing=False)
        
        self._visualize_frame()
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            # eef_site=self._arm.eef_site,
            eef_site=self.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        return np.zeros(6)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [
                0.0,
                -1.5707,
                1.5707,
                -1.5707,
                -1.5707,
                0.0,
            ]
            # put target in a reasonable starting position
            self._target.set_mocap_pose(self._physics, position=[0.5, 0, 0.3], quaternion=[0, 0, 0, 1])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # TODO use the action to control the arm

        # get mocap target pose
        target_pose = self._target.get_mocap_pose(self._physics)

        # run OSC controller to move to target pose
        self._controller.run(target_pose)

        # step physics
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    @property
    def eef_site(self):
        return getattr(self, "_eef_site", self._arm.eef_site)
    
    def _choose_eef_site(self,
                         prefer_gripper_tcp: bool = True,
                         tcp_site_name:str='pinch',
                         create_if_missing: bool = False,
                         tcp_local_pos=(0., 0., 0.145),
                         tcp_local_quat=(1., 0., 0., 0.)):
        
        site = self._arm.eef_site
        
        if prefer_gripper_tcp and hasattr(self, '_gripper'):
            s = self._gripper.mjcf_model.find('site', tcp_site_name)
            if s is None and create_if_missing:
                g_base = self._gripper.mjcf_model.find('body', 'base')
                if g_base is not None:
                    s = g_base.add(
                        "site",
                        name=tcp_site_name,
                        pos=f"{tcp_local_pos[0]} {tcp_local_pos[1]} {tcp_local_pos[2]}",
                        quat=f"{tcp_local_quat[0]} {tcp_local_quat[1]} {tcp_local_quat[2]} {tcp_local_quat[3]}",
                        size="0.008",
                        rgba="0.8 0.2 1 1",
                    )
            if s is not None:
                site = s
        
        self._eef_site = site
        return site
    
    def _sync_target_to_eef(self):
        if self._eef_site is None:
            return
        pos = self._physics.bind(self._eef_site).xpos.copy()
        R = self._physics.bind(self._eef_site).xmat.reshape(3,3).copy()
        quat = mat2quat(R)
        self._target.set_mocap_pose(self._physics, position=pos, quaternion=quat)
        
    def _sync_dbg_goal_axes(self):
        if not hasattr(self, "_dbg_goal_body"):
            return
        x, y, z = self.goal[:3]
        qx, qy, qz, qw = self.goal[3:]
        qpos = _pose_to_freejoint_qpos((x, y, z), (qx, qy, qz, qw))
        self._physics.bind(self._dbg_goal_body.freejoint).qpos = qpos

    def _sync_dbg_cube_axes(self):
        if not hasattr(self, "_dbg_cube_body"):
            return
        # read cube pose from the frame body
        cube_pos = self._physics.bind(self._cube_frame).xpos.copy()
        cube_R   = self._physics.bind(self._cube_frame).xmat.reshape(3,3).copy()
        from manipulator_mujoco.utils.transform_utils import mat2quat
        qx, qy, qz, qw = mat2quat(cube_R)  # xyzw
        qpos = _pose_to_freejoint_qpos(cube_pos, (qx, qy, qz, qw))
        self._physics.bind(self._dbg_cube_body.freejoint).qpos = qpos


    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()
        
        
    def _visualize_frame(self):
        # ----- EEF axes (at the chosen site) -----
        s = getattr(self, "_eef_site", None)
        if s is not None:
            owner_body = s.parent
            pos = _as_vec3(getattr(s, "pos", None))
            quat = _as_quat(getattr(s, "quat", None))
            dbg = owner_body.add("body", name="dbg_eef_axes",
                                pos=f"{pos[0]} {pos[1]} {pos[2]}",
                                quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
            _add_axes_geoms_on_body(dbg, "eef", length=0.08, radius=0.002)

        # ----- TCP (pinch) marker -----
        if hasattr(self, "_gripper"):
            pinch_site = self._gripper.mjcf_model.find("site", "pinch")
            g_base = self._gripper.mjcf_model.find("body", "base")
            if pinch_site is not None and g_base is not None:
                dbg_tcp = g_base.add("body", name="dbg_tcp",
                                    pos="{} {} {}".format(*_as_vec3(pinch_site.pos)),
                                    quat="1 0 0 0")
                dbg_tcp.add("geom", type="sphere", size="0.012",
                            rgba="1 1 0 1", contype="0", conaffinity="0", group="0")

        # ----- Goal axes (free body synced to goal) -----
        if not hasattr(self, "_dbg_goal_body"):
            root = self._arena.mjcf_model.worldbody
            self._dbg_goal_body = root.add("body", name="dbg_goal_body")
            self._dbg_goal_body.add("freejoint", name="dbg_goal_free")
            _add_axes_geoms_on_body(self._dbg_goal_body, "goal", length=0.08, radius=0.002)

        # # ----- Cube axes (free body synced to cube) -----
        # if hasattr(self, "_cube_frame") and not hasattr(self, "_dbg_cube_body"):
        #     root = self._arena.mjcf_model.worldbody
        #     self._dbg_cube_body = root.add("body", name="dbg_cube_body")
        #     self._dbg_cube_body.add("freejoint", name="dbg_cube_free")
        #     _add_axes_geoms_on_body(self._dbg_cube_body, "cube", length=0.06, radius=0.002)



    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()