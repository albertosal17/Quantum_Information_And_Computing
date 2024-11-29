import os

def checkpoint(message, debug = True, **kwargs):
    """
    @brief Prints a checkpoint message and optionally the values of variables.
    
    @param message A string containing the checkpoint message.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.

    @note: arguments for printing the eventually variables should be passed with the following structure:
            Considering for example the case you want to print two variables var1 and var2
            checkpoint("the message", debug=True, var1=var1, var2=var2)

    """
    if debug:
        print(f"[CHECKPOINT] {message}")
        if kwargs:
            for var_name, value in kwargs.items():
                print(f"  {var_name}: {value}")
        print()  # Print a blank line for better readability

def error(error_message):
    """
    @brief Prints an error message and 
    
    @param message A string containing the checkpoint message.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.
    """
    print(f"[ERROR] {error_message}")
    print("Stopping execution.")  
    os._exit(0)

def warning(warning_message, debug=True):
    """
    @brief Prints an error message and 
    
    @param message A string containing the checkpoint message.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.
    """
    if debug:
        print(f"[WARNING] {warning_message}")
    

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

