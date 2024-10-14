import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    
    def __init__(self, dt=0.01, render=True, debug=False):
        """
        Initialize the Unitree A1 environment.

        Args:
            dt (float): Time step for the simulation and CPG updates.
            debug (bool): Whether to enable debug mode.
        """
        super(UnitreeA1Env, self).__init__()

        self.debug = debug

        # Initialize the Pybullet client with robot
        if render:
            client_id = p.connect(p.GUI)
        else:
            client_id = p.connect(p.DIRECT)
        self.robot = UnitreeA1(client=client_id, dt=dt, debug=debug)
<<<<<<< Updated upstream
=======
        self.dt = dt
        self.time_step = 0
        self.max_steps = 1000  # Max steps per episode
        self.discount_factor = 0.99 #Discount factor for rewards
>>>>>>> Stashed changes

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
        
<<<<<<< Updated upstream
        return self._observe()
    
    def close(self):
        """
        Close the environment.
        """
        del(self.robot)
=======
        return observation
    
    # Observation returns [joint_positions, joint_velocities, joint_torques, IMU_data]
    def calculate_reward(self, observation):
        """
        Calculate the reward based on the observation.

        Args:
            observation (np.array): The current observation.

        Returns:
            reward (float): The reward for the current step.
        """
        #Initialize tune-able params, reward weights
        forward_progress_reward_weight = 1.0
        roll_stability_penalty_weight = 1.0
        pitch_stability_penalty_weight = 1.0
        payload_drop_penalty_weight = 1.0
        foot_slip_penalty_weight = 1.0
        max_roll = 0.6 #In radians, around 34 degrees
        min_roll = -0.6
        max_pitch = 0.6
        min_pitch = -0.6
        roll_pitch_range_without_penalty = 0.1745 #10 degrees
        threashold = roll_pitch_range_without_penalty ** 2



        # Extract the info from observations
        ref_joint_positions = observation[0]
        ref_joint_velocities = observation[1]
        ref_base_position = observation[2]
        ref_base_orientation = observation[3]
        ref_foot_velocities = observation[4]
        ref_foot_contacts = observation[5]

    #Calculate the reward based on the observation 
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
        else:
            payload_drop_penalty = 0
        
        # Calculate the penalty for foot slip
        #If the foot has linear velocity while contact is true, then foot slip has occured
        for i in range(4):
            if ref_foot_contacts[i] == True and ref_foot_velocities[i] > 1e-2:
                foot_slip_penalty = 10 * foot_slip_penalty_weight
                break
            else:
                foot_slip_penalty = 0
        
        # Calculate the total reward
        reward = forward_progress_reward - roll_pitch_stability_penalty - payload_drop_penalty - foot_slip_penalty
        return reward


#Calculate the reward and return each reward type
    def calculate_reward_for_testing(self, observation):
        """
        Calculate the reward based on the observation.

        Args:
            observation (np.array): The current observation.

        Returns:
            reward (float): The reward for the current step.
        """
        #Initialize tune-able params, reward weights
        forward_progress_reward_weight = 1.0
        roll_stability_penalty_weight = 1.0
        pitch_stability_penalty_weight = 1.0
        payload_drop_penalty_weight = 1.0
        foot_slip_penalty_weight = 1.0
        max_roll = 0.6 #In radians, around 34 degrees
        min_roll = -0.6
        max_pitch = 0.6
        min_pitch = -0.6
        roll_pitch_range_without_penalty = 0.1745 #10 degrees
        threashold = roll_pitch_range_without_penalty ** 2



        # Extract the info from observations
        ref_joint_positions = observation[0]
        ref_joint_velocities = observation[1]
        ref_base_position = observation[2]
        ref_base_orientation = observation[3]
        ref_foot_velocities = observation[4]
        ref_foot_contacts = observation[5]

        ### Calculate the reward based on the observation ###

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
        else:
            payload_drop_penalty = 0
        
        # Calculate the penalty for foot slip
        #If the foot has linear velocity while contact is true, then foot slip has occured
        for i in range(4):
            if ref_foot_contacts[i] == True and ref_foot_velocities[i] > 1e-2:
                foot_slip_penalty = 10 * foot_slip_penalty_weight
                break
            else:
                foot_slip_penalty = 0
        
        # Calculate the total reward
        reward = forward_progress_reward - roll_pitch_stability_penalty - payload_drop_penalty - foot_slip_penalty
        return reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty

>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
        # Get the reward
        reward = self._get_reward()
=======
        # Calculate the reward
        reward = (self.discount_factor ** self.time_step) * self.calculate_reward(observation)
>>>>>>> Stashed changes

        # Check if the episode is done
        done = self._done()

        if self.robot.debug:
            print(f"[DEBUG] Step reward: {reward}")
            if done:
                print("[DEBUG] Episode done")

        # Addition info (optional)
        info = {}
<<<<<<< Updated upstream
=======
        return observation, reward, done, info
    
    def step_for_reward_testing(self, action):
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

        # Calculate the reward
        reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty = self.calculate_reward_for_testing(observation)

        # Check if the episode is done (e.g., max steps reached)
        self.time_step += 1
        done = self.time_step >= self.max_steps

        if self.robot.debug:
            print(f"[DEBUG] Step: {self.time_step}, Forward Progress Reward: {forward_progress_reward}, Roll_Pitch_Stability_Penalty: {roll_pitch_stability_penalty}, Payload Drop Penalty: {payload_drop_penalty}, Foot Slip Penalty: {foot_slip_penalty}, Reward: {reward}, Done: {done}")

        info = {}
        return observation, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info
>>>>>>> Stashed changes

        return observation, reward, done, info