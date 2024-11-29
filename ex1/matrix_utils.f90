module matrix_utils
    implicit none
contains
    subroutine print_matrix(mat, n)
        !!This is a subroutine to print an N x N matrix of real numbers
        real, dimension(:,:), intent(in) :: mat !intent(in) means that the argument is intended to be read only (impossible to overwrite it)
        !! This is the matrix we are considering
        integer, intent(in) :: n
        integer :: i

        ! Ensure the matrix is square
        if (size(mat, 1) /= n .or. size(mat, 2) /= n) then
            print *, "Error: The matrix must be N x N."
            return
        end if

        ! Loop through each row and print it
        do i = 1, n
            write(*, "(1x, *(f6.2))") mat(i, :) ! Adjust f6.2 as needed
        end do
    end subroutine print_matrix
end module matrix_utils