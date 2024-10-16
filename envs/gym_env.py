import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import config as cfg
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
    - Foot contacts (4 legs)
    """
    def __init__(self):
        self._joint_positions = []
        self._joint_velocities = []
        self._base_position = []
        self._base_orientation = []
        self._foot_velocities = []
        self._foot_contacts = []
    
    def update(self, joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts):
        self._joint_positions = joint_positions
        self._joint_velocities = joint_velocities
        self._base_position = base_position
        self._base_orientation = base_orientation
        self._foot_velocities = foot_velocities
        self._foot_contacts = foot_contacts

    @property
    def all_observations(self):
        return np.concatenate([
            self._joint_positions,
            self._joint_velocities,
            self._base_position,
            self._base_orientation,
            self._foot_velocities,
            self._foot_contacts
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
    @property
    def foot_contacts(self):
        return self._foot_contacts
    

class UnitreeA1Env(gym.Env):
    """
    Custom OpenAI Gym environment for Unitree A1 with CPG control.
    The agent controls the amplitude, while phase and frequency are kept constant.
    """
    
    def __init__(self, dt=cfg.dt, render=cfg.render, debug=cfg.debug, animate_cpg=cfg.animate_cpg):
        """
        Initialize the Unitree A1 environment.

        Args:
            dt (float): Time step for the simulation and CPG updates.
            debug (bool): Whether to enable debug mode.
            animate_cpg (bool): Whether to animate the CPG controller.
        """
        super(UnitreeA1Env, self).__init__()

        self.debug = debug
        self.payload_dropped = False

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
        joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts = self.robot.get_observation()
        self.observation.update(joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts)
        return self.observation
    
    def _get_reward(self, observation: Observations, full_reward=False):
        """
        Calculate the reward based on the observation.

        Args:
            observation (Observations): The current observation.
            full_reward (bool): Whether to return the full reward components.

        Returns:
        if not full_reward:
            reward (float): The reward for the current step.
        if full_reward:
            reward (float): The total reward.
            forward_progress_reward (float): Reward for forward progress.
            roll_pitch_stability_penalty (float): Penalty for roll and pitch stability.
            payload_drop_penalty (float): Penalty for payload drop.
            foot_slip_penalty (float): Penalty for foot slip

        """
        # Initialize tune-able params, reward weights
        # TODO: Set hyperparameters with config file
        forward_progress_reward_weight = cfg.forward_progress_reward_weight
        roll_stability_penalty_weight = cfg.roll_stability_penalty_weight
        pitch_stability_penalty_weight = cfg.pitch_stability_penalty_weight
        payload_drop_penalty_weight = cfg.payload_drop_penalty_weight
        foot_slip_penalty_weight = cfg.foot_slip_penalty_weight
        max_roll = cfg.max_roll
        min_roll = cfg.min_roll
        max_pitch = cfg.max_pitch
        min_pitch = cfg.min_pitch
        threashold = cfg.roll_pitch_penalty_threashold

        # Extract the info from observations
        # ref_joint_positions = observation.joint_positions
        # ref_joint_velocities = observation.joint_velocities
        ref_base_position = observation.base_position
        ref_base_orientation = observation.base_orientation
        ref_foot_velocities = observation.foot_velocities
        ref_foot_contacts = observation.foot_contacts

        ## Calculate the reward based on the observation

        # Calculate the reward for distance travelled from starting position
        forward_progress_reward = forward_progress_reward_weight*((ref_base_position[0]**2 + ref_base_position[1]**2 + ref_base_position[2]**2)**0.5)

        # Calculate the reward for roll and pitch stability
        # Get Roll, Pitch, Yaw from quaternion
        r = R.from_quat(ref_base_orientation)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        roll_stability_penalty = roll_stability_penalty_weight * max(0, roll ** 2 - threashold)
        pitch_stability_penalty = pitch_stability_penalty_weight * max(0, pitch ** 2 - threashold)
        roll_pitch_stability_penalty = roll_stability_penalty + pitch_stability_penalty

        # Calculate the penalty for payload drop
        if roll < min_roll or roll > max_roll or pitch < min_pitch or pitch > max_pitch:
            payload_drop_penalty = 100 * payload_drop_penalty_weight
            self.payload_dropped = True
        else:
            payload_drop_penalty = 0
        
        # Calculate the penalty for foot slip
        # If the foot has linear velocity while contact is true, then foot slip has occured
        for i in range(4):
            if ref_foot_contacts[i] == True and ref_foot_velocities[i] > 1e-2:
                foot_slip_penalty = 10 * foot_slip_penalty_weight
                break
            else:
                foot_slip_penalty = 0
        
        # Calculate the total reward
        reward = forward_progress_reward - roll_pitch_stability_penalty - payload_drop_penalty - foot_slip_penalty
        if full_reward:
            return reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty
        else:
            return reward
    
    def _done(self):
        """
        Check if the episode is done.
        """
        if self.payload_dropped==True:
            return True
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

        # Get the next observation, return custom Observation class
        observation = self._observe()

        # Get the reward
        reward = self._get_reward(observation)

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional)
        info = {}

        return observation, reward, done, info
    
    def step_for_reward_testing(self, action):
        """
        A test function that returns full reward info while stepping.
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

        # Get the next observation, return custom Observation class
        observation = self._observe()

        # Calculate the reward
        reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty = self._get_reward(observation, full_reward=True)

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional)
        info = {}

        return observation, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info