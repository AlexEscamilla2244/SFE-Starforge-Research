import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import numpy as np
from ctypes import c_ubyte
import astropy.units as u
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
from gala.dynamics import mockstream as ms

### Make adjustments here! ###

adjustable_params = {}

adjustable_params['x'] = 20.
adjustable_params['y'] = 50.
adjustable_params['z'] = 30.

adjustable_params['vx'] = -50.
adjustable_params['vy'] = 170.
adjustable_params['vz'] = 0.

adjustable_params['prog_mass'] = 2.5*10**4 * u.Msun
adjustable_params['age'] = 1000

adjustable_params['color'] = 'teal'

### Leave the following code unchanged ###

#Create stream
pot = gp.NFWPotential.from_circular_velocity(v_c=220*u.km/u.s, r_s=15*u.kpc, units=galactic)
H = gp.Hamiltonian(pot)

prog_w0 = gd.PhaseSpacePosition(pos=[adjustable_params['x'], adjustable_params['y'], adjustable_params['z']] * u.kpc,
                                vel=[adjustable_params['vx'], adjustable_params['vy'], adjustable_params['vz']] * u.km/u.s)

df = ms.FardalStreamDF()

gen = ms.MockStreamGenerator(df, H)

my_stream, prog = gen.run(prog_w0, adjustable_params['prog_mass'], dt=1 * u.Myr, n_steps=adjustable_params['age'])

#Plot stream
my_stream.plot(color=adjustable_params['color'],alpha=0.1,s=1.5)
plt.show()

#Uncomment below when you're ready to start saving to disk!

#Enter your name here (for setting the filename):
your_name = 'Alex'

# #Get stream positions
your_stream = np.array(my_stream.pos.xyz)

# #Get stream colors
rgb_color = mpl.colors.to_rgb(adjustable_params['color'])
Npoint = np.size(my_stream.pos.x)
r_array = np.ones(Npoint, dtype=c_ubyte) * int(rgb_color[0] * 255)
g_array = np.ones(Npoint, dtype=c_ubyte) * int(rgb_color[1] * 255)
b_array = np.ones(Npoint, dtype=c_ubyte) * int(rgb_color[2] * 255)
your_colors = np.vstack((r_array, g_array, b_array))

# #Save positions and colors into an HDF5 file
f = h5py.File("{}_stream_{}.h5".format(your_name,adjustable_params['color']), 'w')
position_dset = f.create_dataset("positions", data=your_stream)
color_dset = f.create_dataset("colors", data=your_colors)
f.close()

