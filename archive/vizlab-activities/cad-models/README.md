## CAD models

### Requirements

* numpy
* perlin_numpy (need to install with 'pip install git+https://github.com/pvigier/perlin-numpy')
* pyvista

### Explanation

To make some new CAD models for the VizLab, we're going to be using a simple asteroid generator. This generator samples Perlin noise to build a random asteroid. Perlin noise is a technique for getting noise values that are smooth across dimensions, and it's is a classic approach in the world of procedural generation. To generate the asteroid, we create a Perlin noise cube of a particular size, then choose normally-distributed random values along a sphere to sample the cube with. We then use the noise values at the sampling point to shape the asteroid, adjusting the position on the sphere in or out radially depending on the noise.  

### What next?

Run the provided Python script to generate a starting model. Once you've got that, here are some suggestions for tweaking things in interesting ways:

* Change the noise radius so you're getting different parts of the Perlin noise cube. How does the surface of the asteroid change?
* Adjust the noise scaling factor to change how dramatically the noise shapes the surface of the asteroid. What values looks the best to you?
* If you want to get fancy, consider permuting the set of surface points before it gets passed into pyvista for surface generation/STL export. 