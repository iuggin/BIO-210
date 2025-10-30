import numpy as np
import functions as f
import matplotlib.pyplot as plt
 
patterns = f.generate_patterns(50, 2500)

perturbed_pattern = f.perturb_pattern(patterns[0,],1000)
hebbian_weights = f.hebbian_weights(patterns)
hebbian_dyn = f.dynamics(perturbed_pattern, hebbian_weights, 20)

storkey_weights = f.storkey_weights_vectorized(patterns)
storkey_dyn = f.dynamics(perturbed_pattern, storkey_weights, 20)
hebbian_dyn_async = f.dynamics_async(perturbed_pattern, hebbian_weights, 30000,10000)

storkey_dyn_async = f.dynamics_async(perturbed_pattern, storkey_weights, 30000,10000)
hebbian_dyn = np.array(hebbian_dyn)
storkey_dyn = np.array(storkey_dyn)
hebbian_dyn_async = np.array(hebbian_dyn_async)
storkey_dyn_async = np.array(storkey_dyn_async)

E_hd = np.zeros(hebbian_dyn.shape[0])
for i in range(hebbian_dyn.shape[0]):
    E_hd[i] = f.energy(hebbian_dyn[i,],hebbian_weights)
plt.plot(E_hd)
    
E_sd = np.zeros(storkey_dyn.shape[0])
for i in range(storkey_dyn.shape[0]):
    E_sd[i] = f.energy(storkey_dyn[i,],storkey_weights)
plt.plot(E_sd)


E_hda = np.zeros(hebbian_dyn_async.shape[0])
for i in range(0,hebbian_dyn_async.shape[0],1000):
    E_hda[i] = f.energy(hebbian_dyn_async[i,],hebbian_weights)
plt.plot(E_hda)

E_sda = np.zeros(storkey_dyn_async.shape[0])
for i in range(0, storkey_dyn_async.shape[0],1000):
    E_sda[i] = f.energy(storkey_dyn_async[i,],storkey_weights)
plt.plot(E_sda)
