import numpy as np
import math
from perlin_numpy import generate_perlin_noise_3d
import pyvista as pv

### Make adjustments here! ###

num_samples = 10000
noise_size = 256
noise_scaling_factor = 1
noise_radius = 100 # must be less than or equal to noise_size/2

noise = generate_perlin_noise_3d(
    (noise_size, noise_size, noise_size), (4, 4, 4), tileable=(True, False, False)
)

points = np.zeros((num_samples, 3))
for i in range(num_samples):
    # Get gaussian-nornalized point on sphere
    pos = np.random.normal(size=(3))
    normalized = pos * (1 / math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2))

    # Perturb position based on 3d perlin noise
    indices = np.rint(normalized*noise_radius+np.array((noise_size/2, noise_size/2, noise_size/2)))
    points[i,:] = normalized + noise_scaling_factor*noise[int(indices[0]), int(indices[1]), int(indices[2])]*normalized

## Add additional point adjustments here

# Convert set of points to a mesh
cloud = pv.PolyData(points)
surf = cloud.reconstruct_surface()
# Quick fix to flip normals when they face the interior of the shape
if (np.sum(surf.point_normals[0]) > 0):
    surf = surf.compute_normals(flip_normals=True)
surf.plot()
surf.plot_normals()

#Uncomment below when you're ready to start saving to disk!

# #Enter your name here (for setting the filename):
your_name = 'Alex'

surf.save(your_name + "_asteroid.stl")
