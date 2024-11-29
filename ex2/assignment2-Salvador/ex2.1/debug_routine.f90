module debugger
    implicit none
contains
    subroutine checkpoint(debug, msg)
        ! Declaration of arguments
        logical, intent(in) :: debug        
        character(len=*), optional :: msg   
        
        ! Check if debug is enabled
        if (debug) then
            if (present(msg)) then 
                print *, "Checkpoint:", msg
            else
                print *, "Checkpoint reached."
            end if
        end if
    end subroutine checkpoint
end module debugger
  

!Example of usage:
program main
    use debugger
    implicit none

    ! Print the message as the flag is turned on
    call checkpoint(debug = .TRUE., msg = 'your message')

    !This message will not be printed as the flag is turned off
    call checkpoint(debug = .FALSE., msg = 'your message')
end program main
  