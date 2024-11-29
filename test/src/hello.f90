program hello
    !! This is our program
    implicit none
  
    ! This is just a normal comment
    call say_hello("World!")
  
  contains
  
    subroutine say_hello(name)
      !! Our amazing subroutine to say hello
      character(len=*), intent(in) :: name
        !! Who to say hello to
      write(*, '("Hello, ", a)') name
    end subroutine say_hello
  
  end program hello