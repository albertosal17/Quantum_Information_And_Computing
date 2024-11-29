program variables
    implicit none !this tells the compiler that each variable will be explicitly declared (otherwise the are typed according to the letter the begin with)
    
    integer :: amount
    real :: pi
    complex :: frequency
    character :: initial
    logical :: isOkay !Fortran is case INsensitive
    integer :: age

    !variables assignment (good practice to separate it from declaration)
    amount = 10
    pi = 3.1415927
    frequency = (1.0,-0.5)
    initial = 'A' 
    isOkay = .true. 

    print *, "The value of amount (integer) is:", amount
    print *, "The value of pi (real) is:", pi

    !or read values from keyboard
    print *,'Please enter your age: '
    read (*,*) age
    print  *,'Your age is : ', age
end program variables