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
    



