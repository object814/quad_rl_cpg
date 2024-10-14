import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np

from robots.unitree_a1 import UnitreeA1

class Observations():
    """
    Custom class for managing the state of the environment.
    
    Environment observations:
    - Joint positions (4 legs, 2 joints each)
    - Joint velocities (4 legs, 2 joints each)
    - Base position (x, y, z)
    - Base orientation (quaternion)
    - Foot velocities (4 legs)
    """
    def __init__(self):
        self._joint_positions = []
        self._joint_velocities = []
        self._base_position = []
        self._base_orientation = []
        self._foot_velocities = []
    
    def update(self, joint_positions, joint_velocities, base_position, base_orientation, foot_velocities):
        self._joint_positions = joint_positions
        self._joint_velocities = joint_velocities
        self._base_position = base_position
        self._base_orientation = base_orientation
        self._foot_velocities = foot_velocities

    @property
    def observations(self):
        return np.concatenate([
            self._joint_positions,
            self._joint_velocities,
            self._base_position,
            self._base_orientation,
            self._foot_velocities
        ])
    @property
    def joint_positions(self):
        return self._joint_positions
    @property
    def joint_velocities(self):
        return self._joint_velocities
    @property
    def base_position(self):
        return self._base_position
    @property
    def base_orientation(self):
        return self._base_orientation
    @property
    def foot_velocities(self):  
        return self._foot_velocities
    

class UnitreeA1Env(gym.Env):
    """
    Custom OpenAI Gym environment for Unitree A1 with CPG control.
    The agent controls the amplitude, while phase and frequency are kept constant.
    """
    
    def __init__(self, dt=0.01, render=True, debug=False, animate_cpg=True):
        """
        Initialize the Unitree A1 environment.

        Args:
            dt (float): Time step for the simulation and CPG updates.
            debug (bool): Whether to enable debug mode.
            animate_cpg (bool): Whether to animate the CPG controller.
        """
        super(UnitreeA1Env, self).__init__()

        self.debug = debug

        # Initialize the Pybullet client with robot
        if render:
            client_id = p.connect(p.GUI)
        else:
            client_id = p.connect(p.DIRECT)
        self.robot = UnitreeA1(client=client_id, dt=dt, debug=debug, animate_cpg=animate_cpg)

        # Action space: 12 actions, each can be -1 (decrease) or 1 (increase)
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.int8)

        # Observation space: Concatenated robot observations
        obs_dim = self.robot.get_observation_dimension()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation = Observations() # Custom observation class

    def _act(self, action):
        """
        Apply action to the robot in the simulation and step simulation.
        """
        self.robot.apply_cpg_action(action)
        self.robot.step()
        return True

    def _observe(self):
        """
        Get the current observation from the simulation.
        """
        joint_positions, joint_velocities, base_position, base_orientation, foot_velocities = self.robot.get_observation()
        self.observation.update(joint_positions, joint_velocities, base_position, base_orientation, foot_velocities)
        return self.observation.observations
    
    def _get_reward(self):
        """
        Reward function for the environment.
        """
        joint_positions = self.observation.joint_positions
        joint_velocities = self.observation.joint_velocities
        return 0.0
    
    def _done(self):
        """
        Check if the episode is done.
        """
        return False

    def reset(self):
        """
        Reset the environment to the initial state and return the first observation.
        """
        # Reset simulation and robot
        self.robot.reset()

        if self.debug:
            print(f"[DEBUG] Reset observation")
        
        return self._observe()
    
    def close(self):
        """
        Close the environment.
        """
        del(self.robot)

    def step(self, action):
        """
        Executes an action and returns new state, reward, done, and addition info if needed.

        Args:
            action (np.array): Actions to increase or decrease CPG parameters for each leg.
                - Values in `action` should be -1 (decrease) or 1 (increase).

        Returns:
            observation (np.array): The next observation.
            reward (float): The reward for the current step.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the step.
        """
        # Clip the action values to -1 or 1 with integer type
        action = np.array(action, dtype=np.int8)
        for a in action:
            if a < 0:
                a = -1 # decrease
            else:
                a = 1 # increase

        # Check action shape
        assert action.shape == (12,), f"[ERROR] Invalid action shape. Expected (12,), got {action.shape}"
        
        # Apply the action to the robot and step simulation
        self._act(action)

        # Get the next observation
        observation = self._observe()

        # Get the reward
        reward = self._get_reward()

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional)
        info = {}

        return observation, reward, done, info