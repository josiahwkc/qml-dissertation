import numpy as np

def calculate_target_kernel_alignment(K_matrix, y_labels):
    """
    Calculates the Target Kernel Alignment for a given kernel matrix.
    
    Args:
        K_matrix (np.ndarray): The computed NxN kernel matrix.
        y_labels (np.ndarray): The true labels for the data, formatted as {1, -1}.
        
    Returns:
        float: The alignment score between 0 and 1.
    """
    # Ensure labels are +1 and -1
    y = np.array(y_labels)
    if set(np.unique(y)) == {0, 1}:
        y = 2 * y - 1 
        
    # Create the Ideal Target Matrix (T = y * y^T)
    T_matrix = np.outer(y, y)
    
    # Calculate the Frobenius inner products
    # np.sum(A * B) does element-wise multiplication and sums the result
    inner_KT = np.sum(K_matrix * T_matrix)
    inner_KK = np.sum(K_matrix * K_matrix)
    inner_TT = np.sum(T_matrix * T_matrix)
    
    # Calculate final alignment
    alignment = inner_KT / np.sqrt(inner_KK * inner_TT)
    
    return alignment