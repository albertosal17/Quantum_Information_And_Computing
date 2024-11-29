import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from debugger_module import checkpoint, error, warning

def print_matrix(mat, folder='/home/albertos/quantumInfo/ex4/'):
    """
    @brief Writes the matrix elements to a file in a formatted way.
    
    @param mat The matrix to print (numpy array or sparse matrix).
    @param folder The folder path where the matrix file will be saved.
    
    The output file will contain each element of the matrix along with its position in the format:
    (row, column)    value
    """
    with open(folder+"matrix_print.txt", "w") as file:
        for (i, j), value in np.ndenumerate(mat):
            file.write(f"({i}, {j})\t{value}\n")

def check_square(matrix):
    """
    @brief Check if a given matrix is square.

    @param matrix A numpy.ndarray representing the input matrix to check.

    @return True if the matrix is square, False otherwise.

    @exception ValueError If the input is not a numpy array.

    Example usage:
    @code
    mat = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    result = is_square(mat)  # result is True
    @endcode
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array")

    rows, cols = matrix.shape

    return rows == cols

def gen_diag_simmetric_mat(size, values, sparse, print_mat=False):
    """
    @brief Generates a symmetric matrix from given diagonal values.
    
    @param size The size (number of rows/columns) of the square matrix.
    @param values A list containing either numbers or numpy arrays, representing the elements of the diagonals. 
                  The numbers in the liist represent constant values for the diagonal.
                  The numpy arrays in the list represents a diagonal with elements not necessarely equal to a constant.
    @param sparse Boolean flag to indicate if the matrix should be returned as sparse.
    @param print_mat Boolean flag to indicate if the generated matrix should be printed to a file.
    
    @return A symmetric matrix (sparse or dense based on the 'sparse' parameter).
    
    The input `values` should correspond to the diagonals:
      - The first element represents the main diagonal.
      - Subsequent elements represent diagonals with offsets ±k (e.g., second element for ±1, third for ±2, etc.).

    @note: all the array building up the diagonal should have dimension equal to 'size', even if the diagonal is not
           the main one and contains less elements than 'size'
   
    @note A tri-diagonal matrix (second-order approximation of the kinetic part of the Hamiltonian)
          can be constructed with `values = [2, -1]`. For a third-order approximation 
          (pentadiagonal matrix), use `values = [5/2, -4/3, 1/12]`. Further approximations can be easily implemented
          in the same way.  
    """    

    #PRE-CONDITIONS CHECK:
    # check wether the input array 'values' is not empty
    if len(values)==0:
        error("The 'values' parameter cannot be empty. Provide at least the main diagonal.")
    # Check if values is a single integer (unique diagonal case, with all equal elements), if so convert to a np.array
    if isinstance(values, int):
        values = np.array([values])  # Convert single integer to a numpy array (useful fo using 'enumerate' function)

    #building up the diagonals
    data, offsets = [], []
    #for each diagonal
    for k, val in enumerate(values): 
        #CASE: diagonal with not necessarely constant elements
        if isinstance(val, np.ndarray): 
            if len(val) != size:
                error(f"Diagonal {k} should have {size} elements.")            
            else:
                diagonal = val
        #CASE: diagonal with constant elements
        elif isinstance(val, int):
            diagonal = np.repeat(val, size)
        else:
            error("Error: wrong syntax")
        #filling the diagonal
        data.append(diagonal)
        offsets.append(k)
        # for diagonal other than the main one, fill also the simmetric diagonal with the same elements
        if k > 0:
            data.append(diagonal)
            offsets.append(-k)   
    # generate a sparse matrix with that diagonals
    M = sp.dia_matrix((data, offsets), shape=(size, size))

    if not sparse:
        M = M.toarray() # convert to normal dense matrix

    if print_mat:
        print_matrix(M)

    return M


def diagonalize_hermitian(mat, k, which="SA"): 
    """
    @brief Diagonalizes a matrix and computes a set of its eigenvalues and eigenvectors.
    
    @param mat The input hermitian matrix (numpy array or sparse matrix).
    @param sparse Boolean flag to indicate if the matrix is sparse.
    @param k The number of eigenvalues and eigenvectors to compute.
    @param which Specifies which eigenvalues to compute:
                 "SA" (Smallest Algebraic), "LA" (Largest Algebraic), etc.
                 Default is set to 'SA',i.e., the function will return the eigenvalues and eigenvectors sorted in
                 ascending order starting from the smallest algebraic eigenvalue.
    
    @return A tuple containing:
            - Eigenvalues (sorted in ascending order).
            - Eigenvectors (corresponding to the computed eigenvalues).
    """   
    # PRE CONDITIONS
    # Ensure the matrix is a numpy array or a sparse matrix, and that is hermitian
    tol=1e-10
    if sp.issparse(mat): 
        sparse=True
        difference = mat - mat.getH()
        if not np.all(np.abs(difference.data) < tol):
            error("The input matrix is not hermitian")
    elif isinstance(mat, np.ndarray):
        sparse=False
        if not np.allclose(mat, mat.conjugate().T, atol=tol):
            error("The input matrix is not hermitian")
    else:
        error("Invalid typer for the input matrix. It must be a NumPy array or a SciPy sparse matrix.")

    if sparse:
        eigvals, eigvecs = spl.eigsh(mat, k=k, which=which)   #it return only k eigenvectors
        return eigvals, eigvecs 
    else:
        eigvals, eigvecs = np.linalg.eigh(mat)  #return ALL the eigenvectors
        return eigvals[:k],  eigvecs[:, :k]

    


