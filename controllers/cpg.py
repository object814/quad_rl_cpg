"""
Implementation for CPG controller for robot leg tip trajectory generation.
"""

import numpy as np

class CPG:
    def __init__(self, dt, phase, amplitude=0.01, frequency=3.5, debug=False):
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

        #CPG initial configuration (Useful for resetting the CPG)
        self.ini_amplitude = amplitude
        self.ini_frequency = frequency
        self.ini_phase = phase

        # Internal states for CPG dynamics
        self.current_amplitude = amplitude
        self.current_phase = phase

        # Define parameter limits (TODO: adjust as necessary)
        self.amplitude_min = 0.05
        self.amplitude_max = 0.35
        self.frequency_min = 0.5
        self.frequency_max = 15
        self.phase_min = -np.pi
        self.phase_max = np.pi

        """
        Define robot parameters
            d_step (float): Maximum step length.
            h (float): Robot height.
            gc (float): Maximum swing height.
        """
        self.d_step=0.175
        self.h=0.30
        self.gc=0.10
        self.gp=0.05

        if self.debug:
            print(f"[DEBUG] Initialized CPG with amplitude: {amplitude}, frequency: {frequency}, phase: {phase}")


    def reset(self):
        """
        Reset the CPG parameters to their initial values.
        """
        self.amplitude = self.ini_amplitude
        self.frequency = self.ini_frequency
        self.phase = self.ini_phase
        self.current_amplitude = self.amplitude
        self.current_phase = self.phase

        if self.debug:
            print(f"[DEBUG] CPG reset to amplitude: {self.amplitude}, frequency: {self.frequency}, phase: {self.phase}")


    def update(self, amplitude_delta, frequency_delta, phase_delta):
        """
        Update the CPG parameters and Calculate the foot position based on current CPG state.

        Args:
            amplitude_delta (float): Change in amplitude.
            frequency_delta (float): Change in frequency.
            phase_delta (float): Change in phase.

        Returns:
            tuple: Desired foot position (x, z).    
        """


        #print("(update CPG) BEFORE adding deltas: ")
        #print(f"Deltas: {amplitude_delta}, {frequency_delta}, {phase_delta}")
        #print(f"Amplitude: {self.amplitude}")
        #print(f"Frequency: {self.frequency}")
        #print(f"Phase: {self.phase}")
        #print("--------------------------")

        # Adjust deltas (to 1% of the full range)
        # Originally deltas are just 1 or -1. Here we adjust to
        # +-1% or the whole range.
        amplitude_delta = amplitude_delta * ((self.amplitude_max - self.amplitude_min) *0.01)

        frequency_delta = frequency_delta * ((self.frequency_max - self.frequency_min) *0.01)
        phase_delta = phase_delta * ((self.phase_max - self.phase_min) *0.01)
        self.amplitude = np.clip(self.amplitude + amplitude_delta, self.amplitude_min, self.amplitude_max)
        self.frequency = np.clip(self.frequency + frequency_delta, self.frequency_min, self.frequency_max)
        self.phase += phase_delta
        self.phase = self.phase % (2 * np.pi)  # Phase wrapping
        #print(f"[DEBUG] CPG updated - amplitude: {self.amplitude}, frequency: {self.frequency}, phase: {self.phase}")
        if self.debug:
            print(f"[DEBUG] CPG updated - amplitude: {self.amplitude}, frequency: {self.frequency}, phase: {self.phase}")


        #print("(update CPG) AFTER adding deltas: ")
        #print(f"Amplitude: {self.amplitude}")
        #print(f"Frequency: {self.frequency}")
        #print(f"Phase: {self.phase}")
        #print("--------------------------")
        #print("--------------------------")

        # Update amplitude using dynamic equation
        convergence_rate = 10.0  # Coefficient to control convergence speed
        self.current_amplitude += convergence_rate * (self.amplitude - self.current_amplitude) * self.dt

        # Update phase using frequency and wrap phase
        self.current_phase += self.frequency * self.dt
        self.current_phase = self.current_phase % (2 * np.pi)  # Phase wrapping

        if self.debug:
            print(f"[DEBUG] CPG stepped - current amplitude: {self.current_amplitude}, current phase: {self.current_phase}")

        # Calculate desired x and z positions in the leg's local frame
        x_foot = -self.d_step * (self.current_amplitude - 1) * np.cos(self.current_phase)
        
        # Calculate z position based on phase
        if np.sin(self.current_phase) > 0:
            z_foot = -self.h + self.gc * np.sin(self.current_phase)
        else:
            z_foot = -self.h + self.gp * np.sin(self.current_phase)

        if self.debug:
            print(f"[DEBUG] Foot position - x: {x_foot}, z: {z_foot}")

        return x_foot, z_foot