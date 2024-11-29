
module matmul_methods 
    !! Module containing different possible methods to compute the multiplication of square matrices.
    use debugger_matmul
    implicit none
contains
    function matmul_method1(A, B, dim, debug, verbosity) result(C)
    !! This functions compute the product matrix of two square matrices using the standard algorithms of performing the multiplication "row-by-column".
    !! The elements in position (i,j) of the product matrix is computed as a scalar product between the row i of the left-most matrix and the column j of the right-most one.
        implicit none
        
        ! Declaring arguments and variables to be used
        integer, intent(in):: dim
        !! Matrices dimension
        real, intent(in) :: A(dim,dim) 
        !! The rigth-most matrix in the multiplication.
        real, intent(in) :: B(dim,dim)
        !! The rigth-most matrix in the multiplication.
        logical, intent(in) :: debug   
        !! Flag to be turned on in debugging mode.
        integer, intent(in):: verbosity 
        !! Degree of verbosity in debugging mode. Possible values: 0 (no debug messages), 1 (checkpoints only), 2 (checkpoint and printing matrices if size is appropriate)                         
        real :: C(dim,dim)
        !! The product matrix.
        integer :: i, j, k
        !! Matrix indeces.
        real :: start_time
        !! Initial execution's time.
        real :: end_time
        !! Final execution's time.
        real :: exec_time_1
        !! Overall execution time.

        ! Checking if the multiplication is performable, otherwise stop execution
        call check_matmatmul(debug=debug, matrix1=A, matrix2=B)

        ! Opening the output file where to be written the execution time results
        open(unit=10, file = './data/execution_times_M1.csv', status = 'unknown', position = 'append')   
        ! Initializing valibles
        C = 0.0 

        call cpu_time(start_time)
        ! Performing the multiplication row-by-column
        do i=1, dim   
            do j = 1, dim
                do k = 1, dim
                    C(i,j) = C(i,j) + A(i,k)*B(k,j)
                end do
            end do
        end do
        call cpu_time(end_time)
        exec_time_1 = end_time - start_time
        if (verbosity>0) then
            print *, "Execution Time (seconds): ", exec_time_1
        end if

        ! Writing the results it the output file 
        write(10, '(I6,F15.8,F15.8,F15.8)') dim, exec_time_1
        call checkpoint(debug=debug, msg=" Written to output file")


    end function matmul_method1


    function matmul_method2(A, B, dim, debug, verbosity) result(C)
    !! This functions compute the product matrix of two square matrices using the transpose of the left-moste one.
    !! In this way, it allow to make the product as the sum of scalar products between the columns of the two original matrices.
    !! This is beneficial in Fortran as by default it stores arrays columns in contiguos location of memory, making it easier to retrieve the entries involved in the multiplication.
      
        implicit none

        ! Declaring arguments and variables to be used
        integer, intent(in) :: dim
        !! Matrices dimension
        real, intent(in) :: A(dim,dim) 
        !! The rigth-most matrix in the multiplication.
        real, intent(in) :: B(dim,dim)
        !! The rigth-most matrix in the multiplication.      
        logical, intent(in) :: debug   
        !! Flag to be turned on in debugging mode.
        integer, intent(in):: verbosity 
        !! Degree of verbosity in debugging mode. Possible values: 0 (no debug messages), 1 (checkpoints only), 2 (checkpoint and printing matrices if size is appropriate)           
        real :: A_T(dim,dim)
        !! Transpose of the left-most matrix.
        real :: C(dim,dim)
        !! The product matrix.               
        integer :: i, j, k
        !! Matrix indeces.
        real :: start_time
        !! Initial execution's time.
        real :: end_time
        !! Final execution's time.
        real :: exec_time_2
        !! Overall execution time.

        ! Checking if the multiplication is performable, otherwise stop execution
        call check_matmatmul(debug=debug, matrix1=A, matrix2=B)
        
        ! Opening the output file where to be written the execution time results
        open(unit=20, file ='./data/execution_times_M2.csv', status = 'unknown', position = 'append')   

        ! Initializing variables
        C = 0.0 
        call cpu_time(start_time)
        A_T = transpose(A)
        ! Performing the multiplication column-by-column
        do i=1, dim   
            do j = 1, dim
                do k = 1, dim
                    C(i,j) = C(i,j) + A_T(k,i)*B(k,j) 
                end do
            end do
        end do
        call cpu_time(end_time)
        exec_time_2 = end_time - start_time
        if (verbosity>0) then
            print *, "Execution Time (seconds): ", exec_time_2
        end if

        ! Writing the results it the output file 
        write(20, '(I6,F15.8,F15.8,F15.8)') dim, exec_time_2
        call checkpoint(debug=debug, msg=" Written to output file")

    end function matmul_method2

    function matmul_method3(A, B, dim, debug, verbosity) result(C)
        !! This functions compute the product matrix of two square matrices using the built-in method of Fortran.

        implicit none

        ! Declaring arguments and variables to be used
        integer, intent(in) :: dim
        !! Matrices dimension
        real, intent(in) :: A(dim,dim) 
        !! The rigth-most matrix in the multiplication.
        real, intent(in) :: B(dim,dim)
        !! The rigth-most matrix in the multiplication.       
        logical, intent(in) :: debug   
        !! Flag to be turned on in debugging mode.
        integer, intent(in):: verbosity 
        !! Degree of verbosity in debugging mode. Possible values: 0 (no debug messages), 1 (checkpoints only), 2 (checkpoint and printing matrices if size is appropriate)           
        real :: C(dim,dim)
        !! The product matrix              
        real :: start_time
        !! Initial execution's time.
        real :: end_time
        !! Final execution's time.
        real :: exec_time_3
        !! Overall execution time.

        ! Initializing variables
        C = 0.0 

        ! Opening the output file where to be written the execution time results
        open(unit=30, file = './data/execution_times_M3.csv', status = 'unknown', position = 'append')   

        ! Checking if the multiplication is performable, otherwise stop execution
        call check_matmatmul(debug=debug, matrix1=A, matrix2=B)
        call cpu_time(start_time)
        C = matmul(A,B)
        call cpu_time(end_time)
        exec_time_3 = end_time - start_time
        if (verbosity>0) then
            print *, "Execution Time (seconds): ", exec_time_3
        end if

        ! Writing the results it the output file 
        write(30, '(I6,F15.8,F15.8,F15.8)') dim, exec_time_3
        call checkpoint(debug=debug, msg=" Written to output file")

    end function matmul_method3
end module matmul_methods


program matmatmul
    !! Program that benchmark different methods of performing the matrix multiplication in Fortran, measuring the time it takes to perform it.
    !! You can set different sizes of the square matrices to be multiplied and perform the multiplication multiple times in order to get an average result.

    use matrix_utils
    use debugger_matmul
    use matmul_methods

    implicit none

    ! Declaring variables to be used
    integer, dimension(1) :: dims
    !! Matrix sizes
    real, allocatable :: A(:,:)
    !! The rigth-most matrix in the multiplication.
    real, allocatable :: B(:,:)
    !! The left-most matrix in the multiplication.
    real, allocatable :: C(:,:)
    !! The product matrix.
    integer :: ii
    !! index for looping with the different matrices dimensions.
    integer :: jj
    !! index for looping with the repeated measures of the same configuration.
    integer :: dim 
    !! Specific dimension of the matrices.
    integer :: rep_meas 
    !! Number of repeated measure to be taken for a single configuration.
    integer :: max_printable_size 
    !! Final execution's time.
    logical :: debug   
    !! Flag to be turned on in debugging mode.
    integer :: verbosity 
    !! Degree of verbosity in debugging mode. Possible values: 0 (no debug messages), 1 (checkpoints only), 2 (checkpoint and printing matrices if size is appropriate)


    !Initializing variables
    read *, dims !Choosing the dimension of the matrix
    read *, debug ! Chosing if the debugging mode is to be activated
    read *, verbosity 
    read *, rep_meas ! Chosing the number of repeated measure to be taken 
    max_printable_size = 30 ! Chosing the maximum size of a matrix to be printed in debugging mode
    
    print *
    print *, "Parameter used: "
    print *, "dims =", dims
    print *, "debug =", debug
    print *, "rep_meas =", rep_meas
    print *, "verbosity =", verbosity
    print *
    
    ! Set the seed to be used in generating random entries for input the matrices of the multiplication
    call random_seed()

   ! Loop over different values of the dimension of the matrices
    do ii = 1, size(dims)

        ! Setting the specific dimension to be tested hereafter
        dim = dims(ii)
        call checkpoint(debug=debug, msg=" Dimension set", int_var=dim)

        ! Loop for taking repeated measures of the same configuration
        do jj = 1, rep_meas 

            ! Allocate space in memory based on the current dimension of the matrices
            allocate(A(dim, dim), B(dim, dim), C(dim, dim))

            ! Generating random entries (real numbers between 0 and 1) for matrices A and B 
            call random_number(A)
            call random_number(B)

            if (debug .AND. dim < max_printable_size .AND. verbosity>1) then
                !Print the matrices on the output
                call print_matrix(A, dim)
                call print_matrix(B, dim)
            end if

            ! Checking if the matrices are actually square, otherwise stop execution
            call check_square(debug=debug, matrix=A)
            call check_square(debug=debug, matrix=B)

            call checkpoint(debug=debug, msg="EXECUTING METHOD 1 (row-by-col) ")
            C = matmul_method1(A, B, dim, debug, verbosity)
            ! Check if the product matrix is actually square
            call check_square(debug=debug, matrix=C)
            if (debug .AND. dim < max_printable_size .AND. verbosity>1) then
                !Print the matrix on the output
                call print_matrix(C, dim)
            end if

            call checkpoint(debug=debug, msg="EXECUTING METHOD 2 (col-by-col) ")
            C = matmul_method2(A, B, dim, debug, verbosity)
            ! Check if the product matrix is actually square
            call check_square(debug=debug, matrix=C)
            if (debug .AND. dim < max_printable_size .AND. verbosity>1) then
                !Print the matrix on the output
                call print_matrix(C, dim)
            end if

            call checkpoint(debug=debug, msg="EXECUTING METHOD 3 (built-in) ")
            C = matmul_method3(A, B, dim, debug, verbosity)
            ! Check if the product matrix is actually square
            call check_square(debug=debug, matrix=C)

            if (debug .AND. dim < max_printable_size .AND. verbosity>1) then
                !Print the matrix on the output
                call print_matrix(C, dim)
            end if

            ! Freeing the memory
            deallocate(A,B,C)
        end do
    end do
    print *, "End"
end program matmatmul

