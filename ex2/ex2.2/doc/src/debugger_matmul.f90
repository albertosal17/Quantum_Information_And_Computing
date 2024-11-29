module debugger_matmul
!! This module implements debugging functions useful while performing matrix-matrix multiplications
    implicit none
    
contains

    subroutine checkpoint(debug, msg)
    !! This subroutine is to be used whenever you need a checkpoint in your code, so that you can know that the execution certainly came to that point.
        
        ! Declaration of arguments
        logical, intent(in) :: debug
        !! Flag to be turned on in debugging mode.
        character(len=*), optional :: msg  
        !! The optional message you may want to print at the checkpoint

        ! Check if you are in debugging mode and eventually print a message to communicate you have reached the checkpoint
        if (debug) then
            if (present(msg)) then 
                print *, "Checkpoint:", msg
            else
                print *, "Checkpoint reached."
            end if
        end if
    end subroutine checkpoint

    subroutine fatal_error(debug, msg)
    !! This subroutine is to be used whenever you need to stop the execution because something nasty happened during the execution.
    !! It allows to print a personalizable error-message in debugging mode.

        ! Declaration of arguments
        logical, intent(in) :: debug   
        !! Flag to be turned on in debugging mode.
        character(len=*), optional :: msg  
        !! The optional message you may want to print 
        
        ! Check if you are in debugging mode and eventually print a personalizable error message
        if (debug) then
            if (present(msg)) then 
                print *, "Error:", msg 
            end if
        ! If you are not in debugging mode print a generic error message
        else
            print *, "Fatal error: end of execution"
        end if

        ! Stop the execution
        stop

    end subroutine fatal_error

    subroutine check_square(debug, matrix)
    !! Subroutine that checks wether a matrix is square, i.e., has the same number of columns and rows.

        ! Declaration of arguments
        real, intent(in) :: matrix(:,:)
        !! The matrix to be checked.
        logical, intent(in) :: debug   
        !! Flag to be turned on in debugging mode.

        ! Check if the number of columns is the same as rows, otherwise stop execution
        if (size(matrix,1) /= size(matrix,2)) then 
            call fatal_error(debug=debug, msg="The matrix is not square")
        else 
            call checkpoint(debug=debug, msg="The matrix is square")
        end if
    end subroutine check_square

    subroutine check_matmatmul(debug, matrix1, matrix2)
        !! Subroutine that checks wether a the multiplication between ttwo generic matrices can be done,
        !! i.e., if the number of columns of the left-most one coincides with the number of rows of the right-most one.
    
            ! Declaration of arguments
            real, intent(in) :: matrix1(:,:)
            !! The rigth-most matrix in the multiplication.
            real, intent(in) :: matrix2(:,:)
            !! The left-most matrix in the multiplication.
            logical, intent(in) :: debug   
            !! Flag to be turned on in debugging mode.

            ! Check if the number of columns is the same as rows, otherwise stop execution
            if (size(matrix1, 2) /= size(matrix2, 1)) then 
                call fatal_error(debug=debug, msg="Impossible multiplication")
            else 
                call checkpoint(debug=debug, msg="The matrices can be multiplied as they have appropriate shapes")
            end if
    end subroutine check_matmatmul

end module debugger_matmul
  
