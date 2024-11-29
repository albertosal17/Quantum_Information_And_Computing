module mod_matrix_c8 
!! This module includes a user-defined type and several operations for working with complex matrices in double precision.
    
    type complex8_matrix
    !! Type to be used for storing a complex matrix with double precision elements.
        integer, dimension(2) :: size
        !! The dimensions of the matrix 
        complex*8, dimension(:,:), allocatable :: elem 
        !! The elements of the matrix
    end type 
        
        !Defining the operators associated with this matrix type (based on the modules written below)
        interface operator(.Adj.)
        !! This operator should be used for retreiving the adjoint of the matrix 
            module procedure CMatAdjoint 
        end interface 

        interface operator(.Tr.) 
        !! This operator should be used for retreiving the trace of the matrix 
            module procedure CMatTrace 
        end interface 

    contains 
    !The methods associated with the matrix type
    subroutine  init_complex8_matrix(cmx, rows, cols) 
    !! THis subroutine is to be used to initialize the matrix. By default it is filled with zero entries.
        implicit none

        type(complex8_matrix), intent(out) :: cmx
        !! The complex matrix element to be initialized
        integer, intent(in) :: rows, cols
        !! The dimensions of the matrix

        !Allocating the memory to host the matrix and storing its size
        allocate(cmx%elem(rows,cols))
        cmx%size(1) = rows
        cmx%size(2) = cols

        !Initializing the entries of the matrix
        cmx%elem=(0.0,0.0)
    end subroutine init_complex8_matrix

    function CMatAdjoint(cmx) result(cmxadj) 
    !! This function computes the adjoint (conjugate transpose) of a given complex matrix.
            implicit none
        
            type(complex8_matrix), intent(in) :: cmx 
            !! The original matrix
            type(complex8_matrix) :: cmxadj 
            !! The adjoint matrix

            !Allocating the memory to host the the adjoint matrix.
            !(its dimension are the one of the transpose of the original matrix)
            allocate(cmxadj%elem(cmxadj%size(1),cmxadj%size(2))) 
            cmxadj%size(1) = cmx%size(2)
            cmxadj%size(2) = cmx%size(1) 

            !Computing the adjoint matrix as the conjungate transpose of the original matrix.
            cmxadj%elem = conjg(transpose(cmx%elem)) 
    end function 

    function CMatTrace(cmx) result(tr) 
    !! This function computes the trace of a given complex matrix cmx.
            implicit none

            type(complex8_matrix), intent(in) :: cmx 
            !! The matrix whose trace you want to compute.
            complex*8 :: tr 
            !! The trace of the matrix
            integer :: ii 
            !! Index for do loop 

            !Inizializing variables
            tr = complex(0d0,0d0) ! init to zero before loop 

            !Loop over the main diagonal to compute the trace
            do ii = 1, cmx%size(1) 
               tr = tr + cmx%elem(ii,ii)  !sum of the diagonal elements is the trace by def.
            end do 
    end function 

    subroutine write_matrix_to_file(cmx,  path)
    !! This function is to be used to print the matrix in an output file with appropriate layout.
    !! In particular, each element will be printed with real and imaginary part separated by a withespace.
    !! e.g. if you want to print a 3x3 matrix, in the output you will see a 3x6 matrix
        
        type(complex8_matrix), intent(in) :: cmx
        !! The matrix you want to print
        character(len=*), intent(in) ::  path
        !! The path to the output file where you want to print the matrix

        !open the output file where to print and delete its content if already present
        open(unit=1, file=path, status='replace')

        !print element-by-element the matrix
        do i = 1, cmx%size(1) !for each row
            do j = 1, cmx%size(2) !for each column
                write(1, '(2X, F8.4, 2X, F8.4)', advance='no') real(cmx%elem(i, j)), imag(cmx%elem(i, j))
            end do
            write(1, *)  ! Move to the next line after printing each row
        end do
    end subroutine write_matrix_to_file
end module mod_matrix_c8



