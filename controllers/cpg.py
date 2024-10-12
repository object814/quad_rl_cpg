"""
Implementation for CPG controller for robot leg tip trajectory generation.
"""

import numpy as np

class CPG:
    def __init__(self, dt, amplitude=0.01, frequency=5.0, phase=0.0, debug=False):
        """
        Initialize CPG parameters for a single leg.

        Args:
            dt (float): Time step for updating the states.
            amplitude (float): Initial amplitude value.
            frequency (float): Initial frequency value.
            phase (float): Initial phase value.
            debug (bool): If true, print debug information during updates.
        """
        self.dt = dt
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        self.debug = debug  # Debug flag for printing internal state

        # Internal states for CPG dynamics
        self.current_amplitude = amplitude
        self.current_phase = phase

        # Define parameter limits (example limits, adjust as necessary)
        self.amplitude_min = 0.05
        self.amplitude_max = 0.3
        self.frequency_min = 0.5
        self.frequency_max = 2.5
        self.phase_min = -np.pi
        self.phase_max = np.pi

        if self.debug:
            print(f"[DEBUG] Initialized CPG with amplitude: {amplitude}, frequency: {frequency}, phase: {phase}")

    def reset(self, amplitude=0.01, frequency=5.0, phase=0.0):
        """
        Reset the CPG parameters to their initial values.

        Args:
            amplitude (float): Reset amplitude value.
            frequency (float): Reset frequency value.
            phase (float): Reset phase value.
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.current_amplitude = amplitude
        self.current_phase = phase

        if self.debug:
            print(f"[DEBUG] CPG reset to amplitude: {amplitude}, frequency: {frequency}, phase: {phase}")

    def update(self, amplitude_delta, frequency_delta, phase_delta):
        """
        Update the CPG parameters.

        Args:
            amplitude_delta (float): Change in amplitude.
            frequency_delta (float): Change in frequency.
            phase_delta (float): Change in phase.
        """
        self.amplitude = np.clip(self.amplitude + amplitude_delta, self.amplitude_min, self.amplitude_max)
        self.frequency = np.clip(self.frequency + frequency_delta, self.frequency_min, self.frequency_max)
        self.phase += phase_delta
        self.phase = self.phase % (2 * np.pi)  # Phase wrapping

        if self.debug:
            print(f"[DEBUG] CPG updated - amplitude: {self.amplitude}, frequency: {self.frequency}, phase: {self.phase}")

    def step(self):
        """
        Update the internal CPG states over time.
        """
        # Update amplitude using dynamic equation
        convergence_rate = 10.0  # Coefficient to control convergence speed
        self.current_amplitude += convergence_rate * (self.amplitude - self.current_amplitude) * self.dt

        # Update phase using frequency and wrap phase
        self.current_phase += self.frequency * self.dt
        self.current_phase = self.current_phase % (2 * np.pi)  # Phase wrapping

        if self.debug:
            print(f"[DEBUG] CPG stepped - current amplitude: {self.current_amplitude}, current phase: {self.current_phase}")

    def get_foot_position(self, d_step=0.1, h=0.35, gc=0.05):
        """
        Calculate the desired foot position based on current CPG state.

        Args:
            d_step (float): Maximum step length.
            h (float): Robot height.
            gc (float): Maximum swing height.

        Returns:
            tuple: Desired foot position (x, z).
        """
        # Calculate desired x and z positions in the leg's local frame
        x_foot = -d_step * (self.current_amplitude - 1) * np.cos(self.current_phase)
        
        # Calculate z position based on phase
        if np.sin(self.current_phase) > 0:
            z_foot = -h + gc * np.sin(self.current_phase)
        else:
            z_foot = -h

        if self.debug:
            print(f"[DEBUG] Foot position - x: {x_foot}, z: {z_foot}")

        return x_foot, z_foot
