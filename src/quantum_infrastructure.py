"""
Quantum Backend Provider
============================
Author: Josiah Chan (K23091949)

"""

from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QuantumProvider:
    """
    Manages the execution environment and configuration for quantum circuits.
    """
    
    def __init__(self, shots=1024, use_gpu=False):
        device = 'GPU' if use_gpu else 'CPU'
        
        self.sampler = AerSampler(default_shots=shots)
        self.sampler.options.backend_options = {
            "method": "statevector",
            "device": device
        }
        
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        
        actual_shots = self.sampler.default_shots
        backend_opts = self.sampler.options.backend_options
        actual_device = backend_opts.get("device", "CPU")
        
        print(f"Quantum stack initialised on {actual_device} with {actual_shots} shots.\n")

    def get_kernel(self, feature_map):
        return FidelityQuantumKernel(fidelity=self.fidelity, feature_map=feature_map, max_circuits_per_job=50)