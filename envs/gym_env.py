import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np

from robots.unitree_a1 import UnitreeA1

class UnitreeA1Env(gym.Env):
    """
    Custom OpenAI Gym environment for Unitree A1 with CPG control.
    The agent controls the amplitude, while phase and frequency are kept constant.
    """
    
    def __init__(self, dt=0.01, debug=False):
        """
        Initialize the Unitree A1 environment.

        Args:
            dt (float): Time step for the simulation and CPG updates.
            debug (bool): Whether to enable debug mode.
        """
        super(UnitreeA1Env, self).__init__()

        # Initialize the Pybullet client
        client_id = p.connect(p.GUI)
        p.setTimeStep(dt, physicsClientId=client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Initialize the Unitree A1 robot simulation
        self.robot = UnitreeA1(client=client_id, dt=dt, debug=debug)
        self.dt = dt
        self.time_step = 0
        self.max_steps = 1000  # Max steps per episode

        # Action space: Binary action to increase amplitude, with 4 legs
        # Action space will have 4 legs with binary actions: 0 (do nothing), 1 (increase amplitude)
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Observation space: Concatenated robot observations
        obs_dim = self.robot.get_observation_dimension()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Initial values for amplitude (phase and frequency are constant)
        self.amplitude = np.array([0.01, 0.01, 0.01, 0.01])  # Start with 0.01 amplitude for each leg
        self.amplitude_target = 0.2  # Target amplitude
        self.amplitude_step = 0.01  # Step size for increasing amplitude

    def reset(self):
        """
        Reset the environment to the initial state and return the first observation.
        """
        self.time_step = 0
        self.robot.reset()

        # Reset amplitude to the initial values
        self.amplitude = np.array([0.01, 0.01, 0.01, 0.01])

        # Get the initial observation
        observation = self.robot.get_observation()

        if self.robot.debug:
            print(f"[DEBUG] Reset environment at timestep {self.time_step}")
        
        return observation

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Actions to increase amplitude for each leg.
                - Values in `action` should be 0 (do nothing) or 1 (increase amplitude).

        Returns:
            observation (np.array): The next observation.
            reward (float): The reward for the current step.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the step.
        """
        # Clip the action to ensure valid values (0 for no change, 1 for increase)
        action = np.clip(action, 0, 1)

        # Update amplitudes based on the action, increase only until it reaches 0.2
        for i in range(4):  # 4 legs
            if action[i] == 1 and self.amplitude[i] < self.amplitude_target:
                self.amplitude[i] = min(self.amplitude[i] + self.amplitude_step, self.amplitude_target)

        if self.robot.debug:
            print(f"[DEBUG] Updated amplitudes: {self.amplitude}")

        # Keep frequency and phase constant
        cpg_actions = np.zeros((4, 3))
        cpg_actions[:, 0] = self.amplitude - self.robot.cpg_controllers['FR'].amplitude  # Adjust amplitude
        # Frequency (index 1) and phase (index 2) stay constant (0 adjustment)

        # Apply CPG actions to the robot
        self.robot.apply_cpg_action(cpg_actions)

        # Step the simulation
        self.robot.step()

        # Get the next observation
        observation = self.robot.get_observation()

        # Define a simple reward function (example: penalize large changes in amplitude)
        reward = -np.sum(np.abs(cpg_actions[:, 0]))  # Penalize large changes in amplitude

        # Check if the episode is done (e.g., max steps reached)
        self.time_step += 1
        done = self.time_step >= self.max_steps

        if self.robot.debug:
            print(f"[DEBUG] Step: {self.time_step}, Reward: {reward}, Done: {done}")

        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment (optional, not implemented here).
        """
        pass