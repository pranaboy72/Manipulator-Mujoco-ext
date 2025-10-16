import os
from dm_control import mjcf
from manipulator_mujoco.robots.arm import Arm

_JACOGEN2_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/jaco_gen2/j2n6s300.xml',
)

_JOINTS = (
    # Only specify the main arm joints (exclude gripper joints)
    'j2n6s300_joint_1',
    'j2n6s300_joint_2',
    'j2n6s300_joint_3',
    'j2n6s300_joint_4',
    'j2n6s300_joint_5',
    'j2n6s300_joint_6'
)

_EEF_SITE = 'eef_site'

class JacoGen2(Arm):
    def __init__(self, name: str = None):
        self._mjcf_root = mjcf.from_path(_JACOGEN2_XML)
        if name:
            self._mjcf_root.model = name

        self._joints = [self._mjcf_root.find('joint', name) for name in _JOINTS]

        self._eef_site = self._mjcf_root.find('site', _EEF_SITE)
    
    def attach_tool(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]):
        raise NotImplementedError("JacoGen2 has dedicated gripper, cannot attach tool.")