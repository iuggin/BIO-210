import functions as f 
import numpy as np
# Generate 80 random patterns of size 1000
randm_patterns = f.generate_patterns(80, 1000)
print(f"random patterns: {randm_patterns.shape}")
# Choose a base pattern and perturb 200 of its elements
perturbed_pattern = f.perturb_pattern(randm_patterns[0,],200)
print(f"shape of perturbed pattern: {perturbed_pattern.shape}")
# Run the synchronous update rule until convergence, or up to 20 iterations. Did the network retrieve the original pattern?
weights = f.storkey_weights_vectorized(randm_patterns) # before that we build the hebbian matrix
print(f"weights: {weights.shape}")
output_syn = f.dynamics(perturbed_pattern, weights, 20)
print(f"output syncronous: {len(output_syn)}")
if np.array_equal(output_syn[-1],perturbed_pattern):
    print("The syncr. algorithm retrieved the original pattern")
else:
    print("syncronous did not work")
# Run the asynchronous update rule for a maximum of 20000 iterations, setting 3000 iterations without a change in the state as the convergence criterion. Did the network retrieve the original pattern?
output_asyn = f.dynamics_async(perturbed_pattern, weights, 20000, 3000)
print(f"output asyncronous: {len(output_asyn)}")
if np.array_equal(output_asyn[-1],perturbed_pattern):
    print("The asyncr. algorithm retrieved the original pattern")
else:
    print("asyncronous did not work")

