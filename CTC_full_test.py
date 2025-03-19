import numpy as np
import logging
import time
from scipy.linalg import expm

logging.basicConfig(filename='ctc_full_test.log', level=logging.INFO, 
                    format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CTCFullTest")

# Original constants
RS = 2.0  # Schwarzschild radius
CONFIG = {
    "swarm_size": 5,
    "max_iterations": 200,
    "resolution": 20,  # Matches original scale
    "time_delay_steps": 3,
    "ctc_feedback_factor": 5.0,  # Strong retrocausal effect
    "entanglement_factor": 0.2,
    "charge": 1.0,
    "em_strength": 3.0,
    "nodes": 16  # Spin network nodes
}

TARGET_PHYSICAL_STATE = int(time.time() * 1000)
START_TIME = time.perf_counter_ns() / 1e9
KNOWN_STATE = int(START_TIME * 1000) % 2**32

def repeating_curve(index):
    return 1 if index % 2 == 0 else 0

# Original CTC Spin Network
class SpinNetwork:
    def __init__(self, nodes=CONFIG["nodes"]):
        self.nodes = nodes
        self.edges = [(i, (i + 1) % nodes) for i in range(nodes)]  # Circular topology
        self.state = np.ones(nodes, dtype=complex) / np.sqrt(nodes)

    def evolve(self, H, dt):
        self.state = expm(-1j * H * dt) @ self.state

    def get_adjacency_matrix(self):
        A = np.zeros((self.nodes, self.nodes))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        return A

# Original Tetrahedral Geometry
class CTCTetrahedralField:
    def __init__(self, resolution=CONFIG["resolution"]):
        self.resolution = resolution
        self.coordinates = self._generate_tetrahedral_coordinates()
        self.H = self._build_hamiltonian()

    def _generate_tetrahedral_coordinates(self):
        coords = np.zeros((self.resolution, 4))
        t = np.linspace(0, 2 * np.pi, self.resolution)
        coords[:, 0] = np.cos(t) * np.sin(t)  # x
        coords[:, 1] = np.sin(t) * np.sin(t)  # y
        coords[:, 2] = np.cos(t)              # z
        coords[:, 3] = t / (2 * np.pi)        # t
        return coords

    def _build_hamiltonian(self):
        H = np.zeros((self.resolution, self.resolution), dtype=complex)
        for i in range(self.resolution):
            for j in range(i + 1, self.resolution):
                Δx = self.coordinates[j] - self.coordinates[i]
                distance = np.linalg.norm(Δx)
                if distance > 0:
                    H[i, j] = H[j, i] = 1j / (distance + 1e-10)
        np.fill_diagonal(H, -1j * np.linalg.norm(self.coordinates[:, :3], axis=1))
        return H

    def propagate(self, ψ0, τ):
        return expm(-1j * self.H * τ) @ ψ0

# Original CTC Wormhole Nodes
def generate_wormhole_nodes(resolution=CONFIG["resolution"]):
    nodes = np.zeros((resolution, 4))
    τ = np.linspace(0, 2 * np.pi, resolution)
    R, r = 1.5, 0.5
    ω = 3
    nodes[:, 0] = (R + r * np.cos(ω * τ)) * np.cos(τ)
    nodes[:, 1] = (R + r * np.cos(ω * τ)) * np.sin(τ)
    nodes[:, 2] = r * np.sin(ω * τ)
    nodes[:, 3] = τ / (2 * np.pi)
    return nodes

class SpacetimeSimulator:
    def __init__(self):
        self.resolution = CONFIG["resolution"]
        self.spin_network = SpinNetwork()
        self.tetrahedral_field = CTCTetrahedralField()
        self.wormhole_nodes = generate_wormhole_nodes()
        self.bit_states = np.array([repeating_curve(i) for i in range(self.resolution)], dtype=int)
        self.temporal_entanglement = np.zeros(self.resolution)
        self.quantum_state = np.ones(self.resolution, dtype=complex) / np.sqrt(self.resolution)
        self.history = []
        self.metric = self.compute_metric_tensor()
        # Log initial state
        timestamp = time.perf_counter_ns()
        logger.info(f"Init, Time {timestamp}: Bit States = {self.bit_states.tolist()}")

    def compute_metric_tensor(self):
        metric = np.zeros((self.resolution, 4, 4))
        for i in range(self.resolution):
            x, y, z, t = self.wormhole_nodes[i]
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
        r = np.linalg.norm(self.wormhole_nodes[:, :3], axis=1)
        theta = self.wormhole_nodes[:, 2]
        load_factor = (time.perf_counter_ns() / 1e9 - START_TIME) / 5
        A[:, 0] = CONFIG["charge"] / (4 * np.pi * (r + 1e-8)) * (1 + np.sin(iteration * 0.2) * load_factor)
        A[:, 3] = CONFIG["em_strength"] * r * np.sin(theta) * (1 + load_factor)
        return A

    def quantum_walk(self, iteration, current_time):
        A_mu = self.compute_vector_potential(iteration)
        prob = np.abs(self.quantum_state)**2
        adj_matrix = self.spin_network.get_adjacency_matrix()
        self.spin_network.evolve(adj_matrix, 2 * np.pi / self.resolution)
        for idx in range(self.resolution):
            expected_state = repeating_curve(idx + iteration)
            self.bit_states[idx] = expected_state
            window = prob[max(0, idx - CONFIG["time_delay_steps"]):idx + 1]
            self.temporal_entanglement[idx] = CONFIG["entanglement_factor"] * np.mean(window) if window.size > 0 else 0
            em_perturbation = A_mu[idx, 0] * CONFIG["em_strength"]
            if np.random.random() < abs(em_perturbation) * self.temporal_entanglement[idx]:
                self.bit_states[idx] = 1 - self.bit_states[idx]
        self.quantum_state = self.tetrahedral_field.propagate(self.quantum_state, 2 * np.pi / self.resolution)
        timestamp = time.perf_counter_ns()
        self.history.append((timestamp, self.bit_states.copy()))
        em_effect = np.mean(np.abs(A_mu[:, 0]))
        logger.info(f"Iteration {iteration}, Time {timestamp}: Bit States = {self.bit_states.tolist()}, "
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
                    f"Fitness = {fitness:.2f}, DeltaT = {delta_time:.6f}, CTC Influence = {ctc_influence:.4f}")
        return fitness

    def run(self):
        print("Starting CTC full spin network test...")
        while self.iteration < CONFIG["max_iterations"]:
            current_time = time.perf_counter_ns() / 1e9
            self.simulator.quantum_walk(self.iteration, current_time)
            for particle in self.swarm:
                particle["fitness"] = self.compute_fitness(particle["state"], particle["temporal_pos"])
                particle["state"] = (particle["state"] + repeating_curve(self.iteration)) % 2**32
                particle["temporal_pos"] = current_time
            self.iteration += 1
            time.sleep(0.001)  # 1ms pace
        print("Test complete. Check ctc_full_test.log.")

if __name__ == "__main__":
    solver = PhysicalStateSolver()
    solver.run()
