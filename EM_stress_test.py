import numpy as np
import logging
import time
from scipy.linalg import expm
import os

# Setup logging with nanosecond precision
logging.basicConfig(filename='em_stress_test.log', level=logging.INFO, 
                    format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("EMStressTest")

# Constants
RS = 2.0  # Schwarzschild radius
CONFIG = {
    "swarm_size": 10,        # More particles
    "max_iterations": 1000,  # Longer run
    "resolution": 100,       # Higher resolution for more bit flips
    "time_delay_steps": 3,
    "ctc_feedback_factor": 0.5,
    "entanglement_factor": 0.1,
    "charge": 1.0,
    "em_strength": 1.0
}

# Physical state replacements
TARGET_PHYSICAL_STATE = int(time.time() * 1000)  # Device uptime in ms
START_TIME = time.perf_counter_ns() / 1e9       # Run start in seconds
KNOWN_STATE = int(START_TIME * 1000) % 2**32    # Time-derived reference

# Repeating state curve: 1, 0, 1, 0...
def repeating_curve(index):
    return 1 if index % 2 == 0 else 0

# CTC worldline generator
def generate_ctc(steps=CONFIG["resolution"]):
    τ = np.linspace(0, 2*np.pi, steps)
    worldline = np.zeros((len(τ), 4))
    R, r = 1.5, 0.5
    ω = 3
    worldline[:, 0] = (R + r * np.cos(ω * τ)) * np.cos(τ)
    worldline[:, 1] = (R + r * np.cos(ω * τ)) * np.sin(τ)
    worldline[:, 2] = r * np.sin(ω * τ)
    worldline[:, 3] = τ / (2 * np.pi)
    return worldline

class CTCQubitField:
    def __init__(self, worldline):
        self.worldline = worldline
        self.n = len(worldline)
        self.H = self._build_hamiltonian()

    def _build_hamiltonian(self):
        H = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n):
            j = (i + 1) % self.n
            Δx = self.worldline[j, :3] - self.worldline[i, :3]
            Δt = self.worldline[j, 3] - self.worldline[i, 3]
            interval = Δt**2 - np.dot(Δx, Δx)
            phase = np.sqrt(abs(interval)) * np.sign(interval)
            H[i, j] = H[j, i] = np.exp(-1j * phase)
        np.fill_diagonal(H, -1j * np.linalg.norm(self.worldline[:, :3], axis=1))
        return H

    def propagate(self, ψ0, τ):
        return expm(-1j * self.H * τ) @ ψ0

class SpacetimeSimulator:
    def __init__(self):
        self.resolution = CONFIG["resolution"]
        self.worldline = generate_ctc()
        self.quantum_field = CTCQubitField(self.worldline)
        self.bit_states = np.array([repeating_curve(i) for i in range(self.resolution)], dtype=int)
        self.temporal_entanglement = np.zeros(self.resolution)
        self.quantum_state = np.ones(self.resolution, dtype=complex) / np.sqrt(self.resolution)
        self.history = []
        self.metric = self.compute_metric_tensor()

    def compute_metric_tensor(self):
        metric = np.zeros((self.resolution, 4, 4))
        for i in range(self.resolution):
            x, y, z, t = self.worldline[i]
            r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
            schwarzschild_factor = 1 - RS / r
            metric[i] = np.array([
                [schwarzschild_factor, 1e-6, 0, 0],
                [1e-6, 1 + y**2, 1e-6, 0],
                [0, 1e-6, 1 + z**2, 1e-6],
                [0, 0, 1e-6, -schwarzschild_factor]
            ])
        return metric

    def compute_vector_potential(self, iteration):
        A = np.zeros((self.resolution, 4))
        r = np.linalg.norm(self.worldline[:, :3], axis=1)
        theta = self.worldline[:, 2]
        # Dynamic EM: Vary with iteration and CPU load (approximated by time)
        load_factor = (time.perf_counter_ns() / 1e9 - START_TIME) / 10  # Scale over 10s
        A[:, 0] = CONFIG["charge"] / (4 * np.pi * (r + 1e-8)) * (1 + np.sin(iteration * 0.1) * load_factor)
        A[:, 3] = CONFIG["em_strength"] * r * np.sin(theta) * (1 + load_factor)
        return A

    def quantum_walk(self, iteration, current_time):
        A_mu = self.compute_vector_potential(iteration)
        prob = np.abs(self.quantum_state)**2
        for idx in range(self.resolution):
            expected_state = repeating_curve(idx + iteration)
            self.bit_states[idx] = expected_state
            window = prob[max(0, idx - CONFIG["time_delay_steps"]):idx + 1]
            self.temporal_entanglement[idx] = CONFIG["entanglement_factor"] * np.mean(window) if window.size > 0 else 0
            # Dynamic EM perturbation
            em_perturbation = A_mu[idx, 0] * CONFIG["em_strength"]
            if np.random.random() < abs(em_perturbation) * self.temporal_entanglement[idx]:
                self.bit_states[idx] = 1 - self.bit_states[idx]
        self.quantum_state = self.quantum_field.propagate(self.quantum_state, 2 * np.pi / self.resolution)
        timestamp = time.perf_counter_ns()
        self.history.append((timestamp, self.bit_states.copy()))
        em_effect = np.mean(np.abs(A_mu[:, 0]))  # Average EM perturbation
        logger.info(f"Iteration {iteration}, Time {timestamp}: Bit States = {self.bit_states[:10].tolist()}..., "
                    f"Entanglement = {self.temporal_entanglement[0]:.4f}, EM Effect = {em_effect:.6f}")

class PhysicalStateSolver:
    def __init__(self):
        self.simulator = SpacetimeSimulator()
        self.swarm = [{"state": TARGET_PHYSICAL_STATE + i, "temporal_pos": time.perf_counter_ns() / 1e9} 
                      for i in range(CONFIG["swarm_size"])]
        self.iteration = 0

    def compute_fitness(self, state, temporal_pos):
        current_time = time.perf_counter_ns() / 1e9
        delta_time = current_time - temporal_pos
        base_fitness = abs(state - KNOWN_STATE)
        ctc_influence = 0
        if self.iteration >= CONFIG["time_delay_steps"] and len(self.simulator.history) >= CONFIG["time_delay_steps"]:
            past_states = [h[1] for h in self.simulator.history[-CONFIG["time_delay_steps"]:]]
            ctc_influence = np.mean([s[0] for s in past_states]) * CONFIG["ctc_feedback_factor"]
        fitness = base_fitness + ctc_influence
        logger.info(f"Iteration {self.iteration}, Time {int(current_time * 1e9)}: State = {state}, "
                    f"Fitness = {fitness:.2f}, DeltaT = {delta_time:.6f}")
        return fitness

    def run(self):
        print("Starting EM stress test...")
        while self.iteration < CONFIG["max_iterations"]:
            current_time = time.perf_counter_ns() / 1e9
            self.simulator.quantum_walk(self.iteration, current_time)
            for particle in self.swarm:
                particle["fitness"] = self.compute_fitness(particle["state"], particle["temporal_pos"])
                particle["state"] = (particle["state"] + repeating_curve(self.iteration)) % 2**32
                particle["temporal_pos"] = current_time
            self.iteration += 1
            # Minimal delay to maximize EM activity
            time.sleep(0.0001)  # 100µs
        print("Test complete. Check em_stress_test.log.")

if __name__ == "__main__":
    solver = PhysicalStateSolver()
    solver.run()
