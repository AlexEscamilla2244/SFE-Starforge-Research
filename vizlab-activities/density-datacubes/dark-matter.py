import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py

#Adjust filepath as necessary!
path_to_file = "halos_short.bin"

with open(path_to_file, "rb") as f:
    halos = pickle.load(f, encoding='latin1')

#Notes on some halo properties:
# - ```'mvir'```: halo mass in solar masses
# - ```'rvir'```: halo radius in kiloparsecs
# - ```'x','y','z'```: position in Megaparsecs
# - ```'vx','vy','vz'```: velocity in kilometers/second

# print('The mass of the first halo can be accessed like this:\n')
# print(halos[0]['mvir'])
# print('\n')

# print('The masses of all halos can be accessed like this:\n')
# print(halos['mvir'])

#Visualize cosmic web of halos
plt.figure(figsize=(11,11))

plt.scatter(halos['x'],halos['y'],c='gray',s=0.5,alpha=0.3)

plt.axis('off')
plt.show()

arr = np.vstack((halos['x'], halos['y'], halos['z']))

#Uncomment below when you're ready to start saving to disk!

your_name = 'Alex'
dims = [256, 256, 256]

# # Do the particle binning!
H, edges = np.histogramdd(np.transpose(arr), bins = np.array(dims))

# # Save the resulting dataset
g = h5py.File(your_name+"_darkmatter.h5", 'w')
dset = g.create_dataset("densities", data=H)
g.close()

