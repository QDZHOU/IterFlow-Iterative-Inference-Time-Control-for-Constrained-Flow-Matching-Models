"""D4RL point-maze adapted to official MuJoCo bindings (headless).
Derived from Farama-Foundation/D4RL commit f2a05c0; Apache-2.0.
Changes: native MuJoCo backend and local HDF5 loading; legacy reset/step API.
MuJoCo 3 physics may differ numerically from the paper's MuJoCo 2.0.
"""
import os
from pathlib import Path
import numpy as np
import h5py
import mujoco
from gym import Env, spaces
from .dynamic_mjc import MJCModel

WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(Env):
    def __init__(self):
        self.maze_arr = parse_maze(LARGE_MAZE)
        self._target = np.array(np.argwhere(self.maze_arr == GOAL)[0], dtype=float)
        self.empty_and_goal_locations = list(zip(*np.where(self.maze_arr != WALL)))
        self.name = "maze2d-large-v1"
        self.max_episode_steps = self._max_episode_steps = 800
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        with point_maze(LARGE_MAZE).asfile() as f:
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)
        self.seed(0)

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        loc = self.empty_and_goal_locations[self.np_random.choice(len(self.empty_and_goal_locations))]
        self.set_state(np.array(loc) + self.np_random.uniform(-.1, .1, 2), self.np_random.randn(2) * .1)
        return self._get_obs()

    def step(self, action):
        self.set_state(self.data.qpos.copy(), np.clip(self.data.qvel, -5., 5.))
        self.data.ctrl[:] = np.clip(action, -1., 1.)
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = float(np.linalg.norm(obs[:2] - self._target) <= .5)
        return obs, reward, False, {}

    def get_normalized_score(self, score):
        return (score - 6.7) / (273.99 - 6.7)

    def get_dataset(self):
        path = Path(os.environ.get("D4RL_DATASET_DIR", "datasets")) / "maze2d-large-sparse-v1.hdf5"
        result = {}
        with h5py.File(path, "r") as f:
            for key in ("observations", "actions", "rewards", "terminals", "timeouts"):
                result[key] = f[key][:]
        return result
