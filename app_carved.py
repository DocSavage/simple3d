# Path tracing rendering of a 3d grayscale volume with curved
# cutouts to see how it looks given lighting.

from scene import Scene
import taichi as ti
import taichi.math as tm

import sys
import numpy as np

cube_side = 256

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(0.0, (0.4, 0.6, 0.8))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (0.6, 0.6, 0.6))
scene.set_directional_light((1, 1, 1), 0.2, (0.6, 0.6, 0.6))

# Load 3d grayscale volume where each voxel is a single byte (uint8) in file.
cube_shape = (cube_side, cube_side, cube_side)
filename = f'data/hemibrain_subvol_{cube_side}x{cube_side}x{cube_side}.bin'
try:
    with open(filename, 'rb') as f:
        data = f.read()
    print(f'Loaded {len(data)} bytes from binary file: {filename}')
    np_array = np.frombuffer(data, dtype=np.uint8).reshape(*cube_shape)
except FileNotFoundError:
    sys.exit(f'File {filename} not found.  Exiting...')

@ti.func
def rgb(r, g, b):
    return tm.vec3(r/255.0, g/255.0, b/255.0)

@ti.func
def gray(g):
    return rgb(g, g, g)

@ti.kernel
def load_voxels(bytedata: ti.types.ndarray()):
    # Grayscale volume
    for j, k in ti.ndrange(cube_side, cube_side):
        x = cube_side * tm.sin(j*k/(cube_side*cube_side)*tm.pi) // 1.5
        for i in range(x):
            scene.set_voxel(tm.vec3(i, j, k), 1, gray(bytedata[i, j, k]))
    

load_voxels(np_array)
print('Loaded voxels into scene.')
scene.finish()