import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import config as cfg
from robots.unitree_a1 import UnitreeA1
from .reward_normalization import RewardNormalizer
from .observation_normalization import ObservationNormalizer

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
    def __init__(self, all_observations: np.ndarray = None):
        """
        Initialize with either a full observation numpy array or default empty attributes.

        Args:
            all_observations (np.ndarray, optional): Full observation array with shape (27,).
        """
        if all_observations is not None:
            # Check input shape
            assert all_observations.shape == (27,), f"[ERROR] Invalid observation shape. Expected (27,), got {all_observations.shape}"
            # Extract observations
            self._joint_positions = all_observations[:8]
            self._joint_velocities = all_observations[8:16]
            self._base_position = all_observations[16:19]
            self._base_orientation = all_observations[19:23]
            self._foot_velocities = all_observations[23:27]
            self._foot_contacts = all_observations[27:28]
            self._box_mass = all_observations[28]
            self._box_link = all_observations[29]
        else:
            # Initialize empty lists if no array is provided
            self._joint_positions = []
            self._joint_velocities = []
            self._base_position = []
            self._base_orientation = []
            self._foot_velocities = []
            self._foot_contacts = []
            self._box_mass = 0
            self._box_link = -1
        
    def update(self, joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts, box_mass, box_link):
        self._joint_positions = joint_positions
        self._joint_velocities = joint_velocities
        self._base_position = base_position
        self._base_orientation = base_orientation
        self._foot_velocities = foot_velocities
        self._foot_contacts = foot_contacts
        self._box_mass = box_mass
        self._box_link = box_link   

    @property
    def all_observations(self):
        return np.concatenate([
            self._joint_positions,
            self._joint_velocities,
            self._base_position,
            self._base_orientation,
            self._foot_velocities,
            self._foot_contacts,
            [self._box_mass],
            [self._box_link]
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
    
    def __init__(self, dt=cfg.dt, render=True, debug=False, animate_cpg=False, fixed_base=False, env_add_weight=cfg.add_weight):
        """
        Initialize the Unitree A1 environment.

        Args:
            dt (float): Time step for updating the CPG states, configurate in config.py
            render (bool): Whether to render the simulation, configurate in config.py
            debug (bool): Whether to enable debug mode.
            animate_cpg (bool): Whether to animate the CPGs.
            fixed_base (bool): Whether to fix the base of the robot.
        """
        super(UnitreeA1Env, self).__init__()

        self.debug = debug
        self.payload_dropped = False
        self.last_position = 0
        self.is_weight_enabled = env_add_weight

        # Initialize the Pybullet client with robot
        if render:
            client_id = p.connect(p.GUI)
        else:
            client_id = p.connect(p.DIRECT)
        self.robot = UnitreeA1(client=client_id, dt=dt, debug=debug, animate_cpg=animate_cpg, fixed_base=fixed_base, add_weight=self.is_weight_enabled)

        # Action space: 12 actions, each can be -1 (decrease) or 1 (increase)
        action_dim = self.robot.get_action_dimension()
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.int8)

        # Observation space: Concatenated robot observations
        obs_dim = self.robot.get_observation_dimension()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation = Observations() # Custom observation class

        # Initialize RewardNormalizer
        self.reward_normalizer = RewardNormalizer()

        # Initialize ObservationNormalizer
        self.observation_normalizer = ObservationNormalizer(obs_dim)

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

        Returns:
            observation (np.array): The current observation.
        """
        joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts, box_mass, box_link = self.robot.get_observation()
        self.observation.update(joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts, box_mass, box_link)
        
        # Normalize the observation
        # normalized_observations = self.observation_normalizer.normalize(self.observation.all_observations)
        # return normalized_observations
        return self.observation.all_observations
    
    def _get_reward(self, observation=None, full_reward=False):
        """
        Calculate the reward based on the observation.

        Args:
            observation (array or Observations): The current observation. Accepts either the full Observations instance
                                                 or a flattened array of all observations.
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
        # If `observation` is None, use the environment's current observation
        if observation is None:
            observation = self.observation
        # If observation is an array, convert it to an Observations instance
        if isinstance(observation, np.ndarray):
            observation = Observations(observation) # Initialize from array

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
        forward_progress_reward = forward_progress_reward_weight*(ref_base_position[0] - self.last_position)
        #print(f"forward_progress_reward: {forward_progress_reward}")

        # Calculate the reward for roll and pitch stability
        # Get Roll, Pitch, Yaw from quaternion
        r = R.from_quat(ref_base_orientation)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        roll_stability_penalty = roll_stability_penalty_weight * max(0, abs(roll) - threashold)
        pitch_stability_penalty = pitch_stability_penalty_weight * max(0, abs(pitch) - threashold)
        roll_pitch_stability_penalty = roll_stability_penalty + pitch_stability_penalty
        #print(f"roll_pitch_stability_penalty: {roll_pitch_stability_penalty}")

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
                #print(f"foot_slip_penalty: {foot_slip_penalty}")
                break
            else:
                foot_slip_penalty = 0
        self.last_position = ref_base_position[0]

        # Calculate the total reward
        reward = forward_progress_reward - roll_pitch_stability_penalty - payload_drop_penalty - foot_slip_penalty

        # Normalize the reward
        # normalized_reward = self.reward_normalizer.normalize(reward)
        # if full_reward:
        #     return normalized_reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty
        # else:
        #     return normalized_reward

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
        self.payload_dropped = False
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
        for i in range(len(action)):
            if action[i] < 1e-8:
                action[i] = -1  # decrease
            else:
                action[i] = 1  # increase

        # Check action shape
        assert action.shape == (12,), f"[ERROR] Invalid action shape. Expected (12,), got {action.shape}"
        
        # Apply the action to the robot and step simulation
        self._act(action)

        # Get the next observation, return custom Observation class
        observation = self._observe()

        # Get the reward
        reward = self._get_reward()

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional - empty for now)
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
        reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty = self._get_reward(full_reward=True)

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional)
        info = {}

        return observation, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info