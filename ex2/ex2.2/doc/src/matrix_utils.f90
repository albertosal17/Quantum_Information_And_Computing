module matrix_utils
    !! This module contains subroutine that may be useful dealing with matrices.
    implicit none
contains
    subroutine print_matrix(mat, n)
        !!Print a squared matrix with appropriate layout

        !Declaring arguments and variables to be used
        real, dimension(:,:), intent(in) :: mat
        !! The matrix to be printed
        integer, intent(in) :: n
        !! The dimension of the matrix
        integer :: i

        ! Loop through each row and print it
        do i = 1, n
            write(*, "(1x, *(f6.2))") mat(i, :) ! Adjust f6.2 as needed
        end do

        ! Adding a blank line after printing the matrix
        write(*, *)

    end subroutine print_matrix
end module matrix_utils