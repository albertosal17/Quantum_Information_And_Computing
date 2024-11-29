program test
    use matrix_utils

    implicit none

    integer :: N
    real, allocatable :: A(:,:)

    N=2
    !M=3
    allocate(A(N,N))

    call random_seed()
    call random_number(A)

    call print_matrix(A, N)

    print *, "end"
end program test
