"""
Quantum Feature Map Factory
============================
Author: Josiah Chan (K23091949)

"""

from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile


class FeatureMapFactory:
    """
    Factory for creating transpiled quantum feature maps.
    """
    
    @staticmethod
    def build_zz_map(num_dims, reps, entanglement, sampler):
        """
        Creates and transpiles a ZZFeatureMap for quantum kernel evaluation.
        """
        raw_fm = ZZFeatureMap(
            feature_dimension=num_dims, 
            reps=reps, 
            entanglement=entanglement
        )
        
        # Transpile against the sampler's internal backend
        return transpile(raw_fm, backend=sampler._backend)