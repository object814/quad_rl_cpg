"""
Unitree A1 robot in PyBullet
"""

import pybullet as p
import pybullet_data
import numpy as np
import math

import envs.config as cfg
from controllers.cpg import CPG
from animations.CPGAnimation import CPGAnimation

CONTROLLER_JOINT_INDICES = [3, 4, 7, 8, 11, 12, 15, 16]  # Thigh and calf joints
JOINT_INDEX = {
    "FR": [3, 4],  # thigh, calf
    "FL": [7, 8],
    "RR": [11, 12],
    "RL": [15, 16]
}
END_EFFECTOR_INDICES = [5, 9, 13, 17]  # End-effector indices for each leg
LEG_NAMES = ["FR", "FL", "RR", "RL"] # Leg names

# Inverse kinematics for each leg
def inverse_kinematics_2d(x, y, L1=0.2, L2=0.2):
    """
    Calculate the inverse kinematics for a 2-joint planar robot.

    Parameters:
    x, y : float
        The x and y coordinates of the end-effector in the base frame.
    L1, L2 : float
        Lengths of the first and second links, respectively.

    Returns:
    theta1, theta2 : float
        The joint angles (in radians) for joint 1 and joint 2.
    """

    r = math.sqrt(x ** 2 + y ** 2)
    if r > L1 + L2:
        raise ValueError("Target position is out of reach.")

    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = min(1.0, max(-1.0, cos_theta2))  # Numerical stability
    theta2 = math.acos(cos_theta2)

    k1 = L1 + L2 * cos_theta2
    k2 = L2 * math.sin(theta2)
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    return theta1, theta2


class UnitreeA1:
    def __init__(self, client, dt=cfg.dt, model_pth="assets/a1_description/urdf/a1.urdf", fixed_base=False, reset_position=cfg.reset_position, debug=False, animate_cpg=False, add_weight=cfg.add_weight):
        """
        Initialize Unitree A1 robot in PyBullet.
        
        Args:
            client (int): PyBullet client ID.
            dt (float): Time step for updating the CPG states.
            model_pth (str): Path to the URDF file for the robot model.
            fixed_base (bool): Whether to fix the base of the robot.
            reset_position (list): Initial position of the robot.
            debug (bool): Whether to enable debug mode.
            animate_cpg (bool): Whether to animate the CPGs.
        """
        self.client = client
        self.dt = dt
        self.fixed_base = fixed_base
        self.reset_position = reset_position
        self.debug = debug
        self.animate_cpg = animate_cpg

        p.setTimeStep(dt, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.debug:
            print("[DEBUG] Initializing Unitree A1 robot...")

        self.robot = p.loadURDF(model_pth, self.reset_position, useFixedBase=self.fixed_base, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)

   

        self.leg_names = LEG_NAMES

        if self.animate_cpg:
            # Create an object to for displaying a CPG Animation
            self.animation = CPGAnimation(self.leg_names)

        # Create a CPG controller for each leg with debug flag
        self.cpg_controllers = {leg_name: CPG(dt=self.dt, debug=self.debug) for leg_name in self.leg_names}

        # Initialize joint and link indices
        self.controlled_joint_indices = CONTROLLER_JOINT_INDICES
        self.end_effector_indices = END_EFFECTOR_INDICES

        self.joint_indices = [i for i in range(p.getNumJoints(self.robot, physicsClientId=self.client))]
        self.joint_limits = [
            p.getJointInfo(self.robot, i, physicsClientId=self.client)[8:10] for i in self.controlled_joint_indices
        ]
        self.num_controlled_joints = len(self.controlled_joint_indices)
        self.reset()

    def __del__(self):
        """Close the PyBullet client."""
        p.disconnect(physicsClientId=self.client)
        print("[INFO] PyBullet client disconnected.")

    def reset(self):
        """Reset robot to initial position and reset CPG controllers."""
        if self.debug:
            print("[DEBUG] Resetting robot and CPGs to initial states...")

        # Reset base position and orientation
        p.resetBasePositionAndOrientation(self.robot, self.reset_position, [0, 0, 0, 1], physicsClientId=self.client)

        # Reset all controlled joints to zero position
        for i in self.controlled_joint_indices:
            p.resetJointState(self.robot, i, 0, 0, physicsClientId=self.client)

        # Reset CPG controllers
        for leg in self.cpg_controllers:
            self.cpg_controllers[leg].reset()

        self.add_virtual_weight() #Initiaize the virtual weight at different location every episode reset

    def step(self):
        """Step the simulation."""
        self.apply_virtual_weight_force() #Apply the force everytime step
        p.stepSimulation(physicsClientId=self.client)
        

    def apply_cpg_action(self, cpg_actions):
        """
        Apply actions to CPGs to change the parameters.
        
        Args:
            cpg_actions (numpy.array): Shape of (12,) 
            [FR_amplitude, FR_frequency, FR_phase, FL_amplitude, FL_frequency, FL_phase, RR_amplitude, RR_frequency, RR_phase, RL_amplitude, RL_frequency, RL_phase]
        """
        # Ensure that cpg_actions has the correct shape
        assert cpg_actions.shape == (12,), f"[ERROR] Invalid shape for cpg_actions. Expected (12,), got {cpg_actions.shape}"

        # Initialize an empty dictionary to store the target foot positions for each leg
        target_positions = {}

        for i, leg_name in enumerate(self.leg_names):
            amplitude_delta, frequency_delta, phase_delta = cpg_actions[i * 3: (i + 1) * 3]

            # Update the CPG for this leg and get the new foot position
            foot_position = self.cpg_controllers[leg_name].update(amplitude_delta, frequency_delta, phase_delta)

            # Store the new foot position
            target_positions[leg_name] = foot_position

        # Update the CPG animation with new target positions
        # print("DEBUG, this is target_positions")
        # print(target_positions)
        if self.animate_cpg:
            self.animation.animate(target_positions)

        # Calculate the joint angles using inverse kinematics
        joint_angles = self.compute_joint_angles(target_positions)

        # Apply the joint positions using position control
        self.apply_joint_positions(joint_angles)
        self.step()



    def compute_joint_angles(self, target_positions):
        """
        Compute joint angles using inverse kinematics for each leg based on desired foot positions.

        Args:
            target_positions: A dictionary mapping leg names to their target (x, z) foot positions.

        Returns:
            joint_angles: A dictionary mapping leg names to (thigh_angle, calf_angle) pairs.
        """
        joint_angles = {}
        for leg_name, (x, z) in target_positions.items():
            # Transform the target position to the local frame of the leg
            x_, z_ = -z, -x
            thigh_angle, calf_angle = inverse_kinematics_2d(x_, z_)
            joint_angles[leg_name] = (thigh_angle, calf_angle)
        return joint_angles
    
    def apply_joint_positions(self, joint_angles):
        """
        Apply calculated joint angles to the robot using position control.

        Args:
            joint_angles: Dictionary mapping leg names to (thigh_angle, calf_angle) pairs.
        """
        for leg_name, angles in joint_angles.items():
            thigh_index, calf_index = JOINT_INDEX[leg_name]
            p.setJointMotorControl2(self.robot, thigh_index, p.POSITION_CONTROL, angles[0], force=100, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, calf_index, p.POSITION_CONTROL, angles[1], force=100, physicsClientId=self.client)

    def get_observation(self):
        """
        Return the normalized observation of the robot.
        Returns: numpy array
            joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts
        """
        joint_positions = []
        joint_velocities = []
        foot_velocities = []
        foot_contacts = [False, False, False, False]

        # Collect positions and velocities for controlled joints only
        for i in self.controlled_joint_indices:
            joint_info = p.getJointState(self.robot, i, physicsClientId=self.client)
            # Normalize positions and velocities
            joint_limit = self.joint_limits[self.controlled_joint_indices.index(i)]
            normalized_position = (joint_info[0] - joint_limit[0]) / (joint_limit[1] - joint_limit[0])
            joint_positions.append(normalized_position)
            joint_velocities.append(joint_info[1])

        # Get the base position and orientation
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)

        # Foot tip contact information
        contact_points = p.getContactPoints(self.robot, self.plane, physicsClientId=self.client)
        for contact in contact_points:
            if contact[3] == 5:
                foot_contacts[0] = True
            elif contact[3] == 9:
                foot_contacts[1] = True
            elif contact[3] == 13:
                foot_contacts[2] = True
            elif contact[3] == 17:
                foot_contacts[3] = True

        # Foot tips position and velocity
        for link_index in self.end_effector_indices:
            link_state = p.getLinkState(self.robot, link_index, computeLinkVelocity=True, physicsClientId=self.client)
            link_position = link_state[0]
            linear_velocity = link_state[6] # e.g. (1.0, 1.2, -0.5)
            # Calculate absolute velocity
            linear_velocity = np.linalg.norm(linear_velocity)
            foot_velocities.append(linear_velocity)
        
        # Convert all values to numpy arrays for consistency
        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)
        foot_velocities = np.array(foot_velocities)
        base_position = np.array(base_position)
        base_orientation = np.array(base_orientation)
        foot_contacts = np.array(foot_contacts)

        # Ensure the total observation dimension is correct
        dimension = len(joint_positions) + len(joint_velocities) + len(base_position) + len(base_orientation) + len(foot_velocities) + len(foot_contacts)
        assert dimension == self.get_observation_dimension(), "Observation dimension mismatch"

        return joint_positions, joint_velocities, base_position, base_orientation, foot_velocities, foot_contacts

    def get_observation_dimension(self):
        """Return the dimension of the observation."""
        return 2 * self.num_controlled_joints + 7 + 4 + 4
    
    def get_action_dimension(self):
        """
        Return the dimension of the action.
        """
        return 4 * 3  # 4 legs, 3 parameters per leg (amplitude, frequency, phase)
    
    def add_virtual_weight(self):
        """
        Set up a virtual weight by defining a force vector that will be applied to the robot.
        This force simulates the effect of an added mass without a physical box.
        """
        # Define a random mass for the virtual weight (between 0.5 and 2 kg)
        box_mass = np.random.uniform(0.5, 1.5)
        # box_mass = 2 #For testing virtual force
        self.virtual_force = None
        self.weight_application_link = None

        # Calculate the force vector for gravity in the z-direction
        gravity = -9.81  # Gravity acceleration (m/s^2)
        self.virtual_force = [0, 0, box_mass * gravity]  # Force vector to simulate weight

        # Choose a random link on the robot to apply this force
        self.weight_application_link = np.random.choice([2, 6, 10, 14])

    def apply_virtual_weight_force(self):
        """
        Apply the virtual weight as an external force at each step.
        This function should be called every simulation step to maintain the weight's effect.
        """
        if self.virtual_force is not None and self.weight_application_link is not None:
            p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.weight_application_link,
                forceObj=self.virtual_force,
                posObj=[0, 0, 0],  # Apply at the origin of the specified link
                flags=p.LINK_FRAME  # Apply force in the link's local frame
            )

