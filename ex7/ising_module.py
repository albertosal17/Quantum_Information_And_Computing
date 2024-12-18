import numpy as np

import debugger_module as dbg


def pauli_matrix(choice):

    if choice=='x':
        return np.array([[0,1],
                         [1,0]])
    elif choice=='y':
        return np.array([[0,-1j],
                         [1j,0]])
    elif choice=='z':    
        return np.array([[1,0],
                         [0,-1]]) 
    else:
        dbg.error(f"Invalid input argument for 'choice': should be 'x', 'y', or 'z', while {choice} was passed")



def gen_hamiltonian_field(N, debug=False):

    if N<2:
        dbg.error("N should be at least 2 for a many body system")

    single_spin_contributes=[] 
    for kk in np.arange(1,N+1): #from 1 to N

        matrices = []

        dbg.checkpoint('---------------', debug=debug)
        dbg.checkpoint(f'index: {kk}', debug=debug)

        #Building up the partial tensor product, left side
        if (kk-1) != 0:  # only for indices that are not the first one
            mat_sx = np.eye(2**(kk-1)) 
            dbg.checkpoint(f'mat_sx\n{mat_sx}', debug=debug)

            matrices.append(mat_sx) # append to the list of matrice I will compute the tensor product at the end
        
        # The pauli matrix describing the interaction with the field for the spin at position kk
        sigma_z = pauli_matrix('x')        
        matrices.append(sigma_z) # append to the list of matrice I will compute the tensor product at the end

        #Building up the partial tensor product, right side
        if N-kk != 0: # Only for indices that are not the last one
            mat_dx = np.eye(2**(N-kk))
            dbg.checkpoint(f'mat_dx\n{mat_dx}', debug=debug)

            matrices.append(mat_dx) # append to the list of matrice I will compute the tensor product at the end
        
        # Taking the tensor product of the computed partial tensor products 
        product = np.kron(matrices[0], matrices[1]) #Oss. N>2 ensures that matrices contains at least two elements
        if len(matrices)==3: #if there are more than 2 (i.e. 3) partial contributes to the whole tensor product
            product = np.kron(product, matrices[2])    
        elif len(matrices)>3:
            dbg.error("Something unexpected happened: 'product' should have at most 3 elements: the left-most term, eventually a middle contribute, and a right-most term")

        single_spin_contributes.append(product) # append to the list of tensor products computed for each spin of the system

    #summing all the single spin contributes (tensor products) to get the final term building up the hamiltonian
    if not len(single_spin_contributes)==N:
        dbg.error("Something unexpected happened: we should have a number of terms to sum equal to the number of spins considered")
    field_contribute_to_H=single_spin_contributes[0]
    for jj in np.arange(1, N):
        field_contribute_to_H += single_spin_contributes[jj]
    
    return field_contribute_to_H



def gen_hamiltonian_pairs(N, debug=False):

    if N<2:
        dbg.error("N should be at least 2 for a many body system")

    single_pair_contributes=[] 
    for ii in np.arange(1,N): #from 1 to N-1

        matricess = []

        dbg.checkpoint('---------------', debug=debug)
        dbg.checkpoint(f'index: {ii}', debug=debug)

        #Building up the partial tensor product, left side
        if (ii-1) != 0:  # only for indices that are not the first one
            mat_sx = np.eye(2**(ii-1)) 
            dbg.checkpoint(f'mat_sx\n{mat_sx}', debug=debug)

            matricess.append(mat_sx) # append to the list of matrice I will compute the tensor product at the end
        
        # The pauli matrices describing the interaction with the field for the spin at position ii
        sigma_x_i = pauli_matrix('z')    
        sigma_x_i_plus_one = pauli_matrix('z')    
        # append to the list of matrice I will compute the tensor product at the end
        matricess.append(sigma_x_i)
        matricess.append(sigma_x_i_plus_one)   
        

        #Building up the partial tensor product, right side
        if  N-(ii+1) != 0: # Only for indices that are not the previous to last one (NOTE: We are ranging from 1 to N-1)
            mat_dx = np.eye( 2**( N-(ii+1) ) )
            dbg.checkpoint(f'mat_dx\n{mat_dx}', debug=debug)

            matricess.append(mat_dx) # append to the list of matrice I will compute the tensor product at the end
        
        # Taking the tensor product of the computed partial tensor products 
        productt = np.kron(matricess[0], matricess[1]) #Oss. N>2 ensures that matrices contains at least three elements
        productt = np.kron(productt, matricess[2]) 
        if len(matricess)==4: #if there are more than 2 (i.e. 3) partial contributes to the whole tensor product
            productt = np.kron(productt, matricess[3])    
        elif len(matricess)>4:
            dbg.error("Something unexpected happened: 'product' should have at most 4 elements: the left-most term, the two middle contributes, and a right-most term")

        single_pair_contributes.append(productt) # append to the list of tensor products computed for each spin of the system

    if not len(single_pair_contributes) == (N-1):
        dbg.error("Something unexpected happened: we should have a number of terms to sum equal to the number of spins considered")
    #summing all the single spin contributes (tensor products) to get the final term building up the hamiltonian    
    pairs_contribute_to_H=single_pair_contributes[0]
    for hh in np.arange(1, N-1):
        pairs_contribute_to_H += single_pair_contributes[hh]
    
    return pairs_contribute_to_H


def hamiltonian_ising(N, field_strength, debug=False, print_hamiltonian=False):
    H_field = gen_hamiltonian_field(N, debug)
    H_pairs = gen_hamiltonian_pairs(N, debug)
    H = field_strength*H_field + H_pairs

    dbg.checkpoint(H.shape, debug=True)

    if print_hamiltonian:
        np.set_printoptions(
                precision=2,      # Number of decimal places
                suppress=True,     # Avoid scientific notation for small numbers
                linewidth=120      # Total width of the output line
            )
        print(H)
    
    return H

