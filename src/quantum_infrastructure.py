from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QuantumProvider:
    def __init__(self, shots=1024, use_gpu=False):
        self.kernel = None
        
        device = 'GPU' if use_gpu else 'CPU'
        self.backend = AerSimulator(method='statevector', device=device)
        
        self.sampler = AerSampler(default_shots=shots)
        
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        print(f"Full Aer Stack initialized on {device} with {shots} shots.\n")

    def get_kernel(self, feature_map):
        """Returns a FidelityQuantumKernel using the full Sampler stack."""
        return FidelityQuantumKernel(fidelity=self.fidelity, feature_map=feature_map)