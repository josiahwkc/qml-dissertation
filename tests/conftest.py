"""
Global Pytest Configuration and Mocking
This file runs automatically before any tests to stub out heavy dependencies.
"""

import sys
import os
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _pkg in [
    "qiskit", "qiskit.circuit", "qiskit.circuit.library",
    "qiskit_aer", "qiskit_aer.primitives",
    "qiskit_machine_learning", "qiskit_machine_learning.datasets",
    "qiskit_machine_learning.kernels", "qiskit_machine_learning.state_fidelities",
]:
    if _pkg not in sys.modules:
        _make_stub(_pkg)

# Provide the symbols actually imported by production modules
sys.modules["qiskit"].transpile = MagicMock(name="transpile", side_effect=lambda circuit, **kw: circuit)
sys.modules["qiskit.circuit.library"].ZZFeatureMap = MagicMock(name="ZZFeatureMap")
sys.modules["qiskit_machine_learning.kernels"].FidelityQuantumKernel = MagicMock()
sys.modules["qiskit_machine_learning.kernels"].FidelityStatevectorKernel = MagicMock()
sys.modules["qiskit_machine_learning.state_fidelities"].ComputeUncompute = MagicMock()
sys.modules["qiskit_aer"].AerSimulator = MagicMock()
sys.modules["qiskit_aer.primitives"].SamplerV2 = MagicMock()
sys.modules["qiskit_machine_learning.datasets"].ad_hoc_data = MagicMock()