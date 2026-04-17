from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile

class FeatureMapFactory:
    @staticmethod
    def build_zz_map(num_dims, reps, entanglement, sampler):
        """
        Creates and transpiles a ZZFeatureMap. 
        Transpiling here prevents AlgorithmErrors during the experiment loops.
        """
        raw_fm = ZZFeatureMap(
            feature_dimension=num_dims, 
            reps=reps, 
            entanglement=entanglement
        )
        
        # Transpile against the sampler's own internal backend
        return transpile(raw_fm, backend=sampler._backend)