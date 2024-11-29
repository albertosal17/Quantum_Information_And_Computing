
program matmatmul
    use matrix_utils
    implicit none

    !!! METHOD 1 [standard row by column]

    !Declaration of variables
    !integer, dimension(8) :: N_values = [10, 100, 500, 800, 1000, 1500, 2000, 2500]
    integer, dimension(5) :: N_values = [4000,5000,6000,8000,10000]
    real, allocatable :: A(:,:), B(:,:), C(:,:), A_T(:,:)
    integer :: ii, jj, N !for looping through different matrix sizes
    real :: start_time, end_time, exec_time_1, exec_time_2, exec_time_3

    open(unit=1, file = 'data.csv', status = 'unknown', position = 'append')   
    !In Fortran, a unit number is an integer identifier that serves as a reference for input and output operations (I/O) with files or other devices. Each unit number corresponds to a specific file or device that you are reading from or writing to.
    !The 'replace' option will overwrite the file if it exists. 

    call random_seed()

    exec_time_1 = -9999.0
    exec_time_2 = -9999.0

   ! Loop over different values of N
    do ii = 1, size(N_values)
        print *, N_values(ii)
        do jj = 1, 4 !4 misure ripetute della stessa configurazione

            !Initialization of variables
            N = N_values(ii)
            
            ! Allocate matrices based on the current value of N
            allocate(A(N, N), B(N, N), C(N, N), A_T(N, N))

            !generating random entries (real numbers between 0 and 1) for matrices A and B 
            call random_number(A)
            call random_number(B)

            !!! METHOD 3: [Built-in function matmul]
            C = 0.0 

            call cpu_time(start_time)
            C = matmul(A,B)
            call cpu_time(end_time)
            exec_time_3 = end_time - start_time
            print *, "METHOD 3 (built-in): Execution Time (seconds): ", exec_time_3

            !call print_matrix(C, N)

            !output data into a file 
            write(1, '(I6,F15.8,F15.8,F15.8)') N, exec_time_1, exec_time_2, exec_time_3

            deallocate(A,B,C,A_T)
        
        end do
        
    end do
  

end program matmatmul