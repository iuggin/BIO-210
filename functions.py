# %%
import numpy as np

# %%
def generate_patterns(num_patterns, pattern_size):
    W = np.where(np.random.randn(num_patterns,pattern_size)<0,-1,1)
    return W

def perturb_pattern(pattern, num_perturb):
    index = np.random.choice(pattern.size, num_perturb, replace=False)
    index = index.astype(int)
    p0 = pattern.copy()
    for i in range(num_perturb):
        p0[index[i]] = p0[index[i]]*(-1)
    return p0
    
def pattern_match(memorized_patterns, pattern):
    m = memorized_patterns.shape[0]
    for i in range(m):
        if np.array_equal(memorized_patterns[i,], pattern):
            return i

def hebbian_weights(patterns):
    M,N = patterns.shape
    W_ij = np.zeros((N,N))
    for h in range(M):
        for j in range(N):
            for i in range(N):
                W_ij[j,i] += patterns[h,j]*patterns[h,i]
    W_ij = 1/M*W_ij
    for i in range(N):
        W_ij[i,i] = 0 
    return W_ij

def update(state, weights):
    p_ti = weights@state
    for j in range(len(p_ti)):
        if p_ti[j]>0:
            p_ti[j]=1
        else:
            p_ti[j]=-1
    return p_ti

def update_async(state,weights):
    M = weights.shape[0]
    rand_row = np.random.randint(0,M)
    p_t1 = weights[rand_row,]@state
    if p_t1>=0:
        p_t1=1
    else:
        p_t1=-1
    state[rand_row]=p_t1
    return state

def dynamics(state, weights, max_iter):
    old_state = np.zeros(len(state))
    output = []
    for i in range(max_iter):
        state = np.dot(weights, state.copy())
        state = np.where(state>=0,1,-1)
        output.append(state.copy())
        if np.array_equal(state, old_state):
            break
        else:
            old_state = state.copy()
    return output



# %%
def dynamics_async(state, weights, max_iter, convergence_num_iter):
    counter = 0
    M, N = weights.shape()
    rand_int = np.random.randint(0,M)
    old_state = np.zeros(len(state))
    output = []
    for i in range(max_iter):
        state = weights[rand_int,]@state
        state = np.where(state>=0,1,-1)
        output.append(state.copy())
        if np.array_equal(state, old_state):
            break
        else:
            old_state = state.copy()
    return output

def dynamics_async(state, weights, max_iter, convergence_num_iter):
    output = []
    counter = 0
    M,N = weights.shape
    old_state = np.zeros(N)
    updated_neuron = 0
    output.append(state.copy())

    for i in range(max_iter):
        rand_int = np.random.randint(0,M)   
        updated_neuron = np.where(np.dot(weights[rand_int,],state)<0,-1,1)
        old_state = state.copy()
        state[rand_int] = updated_neuron
        output.append(state.copy())
        if np.array_equal(old_state, state):
            counter += 1
            if counter == convergence_num_iter:
                break
        else:
            counter = 0
    return output
import numpy as np
def storkey_weights_vectorized(patterns):
    num_patterns, pattern_size = patterns.shape
    
    # 1. Learning pattern 1 (Vectorized)
    p1 = patterns[0]
    W = (1 / pattern_size) * np.outer(p1, p1)
    np.fill_diagonal(W, 0)  # Set diagonal to zero

    # 2. Learning subsequent patterns (Vectorized)
    for y in range(1, num_patterns):
        p_y = patterns[y]
        
        # Calculate H-matrix (O(N^2) instead of O(N^3))
        h = W.dot(p_y)
        H = h[:, np.newaxis] - (W * p_y)
        np.fill_diagonal(H, 0)
        
        # Calculate W update (Vectorized)
        term1 = np.outer(p_y, p_y)
        term2 = p_y[:, np.newaxis] * H.T
        term3 = p_y * H
        
        delta_W = (1 / pattern_size) * (term1 - term2 - term3)
        
        # Apply update only to off-diagonal
        np.fill_diagonal(delta_W, 0)
        W += delta_W
        
    return W
def storkey_weights(patterns):

    num_patterns, pattern_size = patterns.shape

    # --- Part 1: Vectorized with np.outer ---
    p1 = patterns[0]
    W = (1 / pattern_size) * np.outer(p1, p1)
    np.fill_diagonal(W, 0) # Set diagonal to zero

    # learning pattern 2
    for y in range(1, num_patterns):
        
        # --- Part 2: H-matrix (Left as-is) ---
        H = np.zeros((pattern_size, pattern_size))  # calculate H matrix
        for i in range(pattern_size):
            for j in range(pattern_size):
                if i != j:
                    for k in range(pattern_size):
                        if k != i and k != j:
                            H[i,j] += W[i,k]*patterns[y,k]
                            
        # --- Part 3: Update (Partially vectorized) ---
        p_y = patterns[y]
        # Pre-calculate the outer product term
        term1_matrix = np.outer(p_y, p_y) 
        
        for i in range(pattern_size):
            for j in range(pattern_size):
                if i != j:
                    # Look up the value from the pre-calculated matrix
                    delta = (term1_matrix[i,j] - 
                             p_y[i]*H[j,i] - 
                             p_y[j]*H[i,j])
                             
                    W[i,j] = W[i,j] + (1 / pattern_size) * delta
    return W

def energy(state, weights):
    E = 0
    for i in range(len(state)):
        for j in range(len(state)):
            E += weights[i,j] * state[i] * state[j]
    
    return -E/2