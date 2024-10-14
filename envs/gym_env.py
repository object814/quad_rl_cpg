import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        discount_factor = 0.99
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
        return reward




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

        # Calculate the reward
        reward = self.calculate_reward(observation)

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
            print(f"[DEBUG] Step: {self.time_step}, Reward: {reward}, Done: {done}")

        info = {}
        return observation, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info
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
