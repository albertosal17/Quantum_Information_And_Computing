program sum_comparison
    implicit none
    
    !Declaration of variables
    integer*2 :: n1_short !2-bytes integer
    integer*2 :: n2_short
    integer*2 :: result_short
    integer*4 :: n1_long  !4-bytes integer
    integer*4 :: n2_long   
    integer*4 :: result_long   

    !> Single precision real numbers, 6 digits, range 10⁻³⁷ to 10³⁷-1; 32 bits
    integer, parameter :: sp = selected_real_kind(6, 37)
    !> Double precision real numbers, 15 digits, range 10⁻³⁰⁷ to 10³⁰⁷-1; 64 bits
    integer, parameter :: dp = selected_real_kind(15, 307)
    !pi
    real(4), parameter :: PI_4 = 4 * atan (1.0_4) !single precision
    real(8),  parameter :: PI_8  = 4 * atan (1.0_8)  !double precision
    ! Variables for single precision
    real(sp) :: a_sp, b_sp, sqrt2_sp, sum_sp
    ! Variables for double precision
    real(dp) :: a_dp, b_dp, sqrt2_dp, sum_dp



    !!! SUM OF INTEGERS

    n1_short=2000000
    n2_short=1
    n1_long=2000000
    n2_long=1

    result_short = n1_short+n2_short
    result_long = n1_long+n2_long

    print *, "The sum of 2000000 and 1 with 2-byte precision is: ", result_short
    print *, "The sum of 2000000 and 1  with 4-byte precision is: ", result_long


    !!!SUM OF REALS

    a_sp=1.0e32 !10^32
    b_sp=1.0e21 !10^21
    sqrt2_sp = sqrt(2.0_sp) !!real(4) single precision definition

    a_dp=1.0d32 
    b_dp=1.0d21
    sqrt2_dp = dsqrt(2.0_dp) !!real(8) double precision definition

    sum_sp = PI_4*a_sp + sqrt2_sp*b_sp
    sum_dp = PI_8*a_dp + sqrt2_dp*b_dp

    print *
    print *, "The sum of pi*10^32 and sqrt(2)*10^21 with single precision is: ", sum_sp
    print *, "The sum of pi*10^32 and sqrt(2)*10^21 with single precision is: ", sum_dp



end program sum_comparison