program test_complex_matrix
!!The aim of this program is to test the capabilities of the user-defined type described in the 'mod_matrix_c8' module
    use mod_matrix_c8
    implicit none

    ! Declaring variables
    type(complex8_matrix) :: mat, mat_dagger

    ! Initializing the matrix (arbitrary choice of the values)
    call init_complex8_matrix(mat, rows=3, cols=3) !By default now it is initialized to zero
    mat%elem(1,1) = cmplx(1.0, -2.0)  
    mat%elem(1,2) = cmplx(5.0, -1.0)  
    mat%elem(2,2) = cmplx(12.0, 0.0)  
    mat%elem(3,3) = cmplx(9.0, -3.0)  
    mat%elem(3,1) = cmplx(1.0, 10.0)  

    !Print the matrix in an output file
    call write_matrix_to_file(cmx=mat, path="./cmx.txt")

    ! Computing and printing the trace of the matrix
    print *, "Trace of the matrix: ", .Tr. mat

    ! Computing the adjoint matrix
    mat_dagger = .Adj. mat
    
    !Print the adjoint matrix in a different output file
    call write_matrix_to_file(cmx=mat_dagger, path="./cmx_dag.txt")
end program test_complex_matrix