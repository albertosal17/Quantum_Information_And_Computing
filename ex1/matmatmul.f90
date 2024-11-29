
program matmatmul
    !!This is the main program 
    use matrix_utils
    implicit none

    !!! METHOD 1 [standard row by column]

    !Declaration of variables
    !integer, dimension(8) :: N_values = [10, 100, 500, 800, 1000, 1500, 2000, 2500]
    integer, dimension(3) :: N_values = [4000, 5000, 6000]
    real, allocatable :: A(:,:), B(:,:), C(:,:), A_T(:,:)
    integer :: i,j,k !matrices index
    integer :: ii, jj, N !for looping through different matrix sizes
    real :: start_time, end_time, exec_time_1, exec_time_2, exec_time_3

    open(unit=2, file = 'data_optimized.csv', status = 'unknown', position = 'append')   
    !In Fortran, a unit number is an integer identifier that serves as a reference for input and output operations (I/O) with files or other devices. Each unit number corresponds to a specific file or device that you are reading from or writing to.
    !The 'replace' option will overwrite the file if it exists. 

    call random_seed()

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

            !!! METHOD 1: [row by col] 

            C = 0.0 !Product matrix to be filled

            call cpu_time(start_time)
            !Performing the multiplication row-by-column
            do i=1, N   
                do j = 1, N
                    do k = 1, N
                        C(i,j) = C(i,j) + A(i,k)*B(k,j)
                    end do
                end do
            end do
            call cpu_time(end_time)

            exec_time_1 = end_time - start_time
            print *, "METHOD 1 (row-by-col): Execution Time (seconds): ", exec_time_1
            !call print_matrix(C, N)



            !!! METHOD 2: [col-by-col (transposing A)] 
            !As matrices in FOrtran are stored by column, it is convenient to redifine the product as a col by col with the transposed matrix of A
            C = 0.0 

            A_T = transpose(A)
            !call print_matrix(B, N)
            !print * !new line
            !call print_matrix(B_T, n)

            call cpu_time(start_time)
            !Performing the multiplication row-by-column
            do i=1, N   
                do j = 1, N
                    do k = 1, N
                        C(i,j) = C(i,j) + A_T(k,i)*B(k,j) !Notice that indices are inverted
                    end do
                end do
            end do
            call cpu_time(end_time)

            exec_time_2 = end_time - start_time
            print *, "METHOD 2 (transpose A): Execution Time (seconds): ", exec_time_2

            !call print_matrix(C, N)

            !!! METHOD 3: [Built-in function matmul]
            C = 0.0 

            call cpu_time(start_time)
            C = matmul(A,B)
            call cpu_time(end_time)
            exec_time_3 = end_time - start_time
            print *, "METHOD 3 (built-in): Execution Time (seconds): ", exec_time_3

            !call print_matrix(C, N)

            !output data into a file 
            write(2, '(I6,F15.8,F15.8,F15.8)') N, exec_time_1, exec_time_2, exec_time_3

            deallocate(A,B,C,A_T)
        
        end do
        
    end do
  

end program matmatmul