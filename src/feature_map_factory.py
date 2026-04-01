from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile

class FeatureMapFactory:
    @staticmethod
    def build_zz_map(num_dims, reps, entanglement, backend):
        """
        Creates and transpiles a ZZFeatureMap. 
        Transpiling here prevents AlgorithmErrors during the experiment loops.
        """
        raw_fm = ZZFeatureMap(
            feature_dimension=num_dims, 
            reps=reps, 
            entanglement=entanglement
        )
        # Convert abstract ZZ gates into simulator-friendly gates once
        return transpile(raw_fm, backend=backend)