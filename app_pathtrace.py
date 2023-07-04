# Path tracing rendering of a 3d grayscale volume with cutouts to
# show a path through the volume.

from scene import Scene
import taichi as ti
import taichi.math as tm

import sys
import random
import numpy as np

cube_side = 256

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(0.0, (0.4, 0.6, 0.8))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, 0), 0.2, (0.6, 0.6, 0.6))
# scene.set_directional_light((1, 1, 1), 0.2, (0.6, 0.6, 0.6))
# scene.set_directional_light((1, -1, 1), 0.2, (0.6, 0.6, 0.6))
# scene.set_directional_light((-1, 1, 1), 0.2, (0.6, 0.6, 0.6))
# scene.set_directional_light((-1, -1, 1), 0.2, (0.6, 0.6, 0.6))
# scene.set_directional_light((-1, 1, -1), 0.2, (0.6, 0.6, 0.6))

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

@ti.func
def show_voxel(i, j, k):
    show = False
    if i < cube_side // 2 or j < cube_side // 2:
        show = True
    return show

@ti.kernel
def load_voxels(bytedata: ti.types.ndarray(), pathfield: ti.template()):
    # Grayscale volume
    for i, j, k in ti.ndrange(cube_side, cube_side, cube_side):
        if i == pathfield[k, 0] and j == pathfield[k, 1]:
            scene.set_voxel(tm.vec3(i, j, k), 2, gray(bytedata[i, j, k]))
        elif i < pathfield[k, 0] or j < pathfield[k, 1]:
            scene.set_voxel(tm.vec3(i, j, k), 1, gray(bytedata[i, j, k]))
    

pathfield = ti.field(int, shape=(cube_side, 2))
i = cube_side // 2
j = cube_side // 2
perturb = 2
for k in range(cube_side):
    i += random.randint(-perturb, perturb)
    j += random.randint(-perturb, perturb)
    pathfield[k, 0] = i
    pathfield[k, 1] = j

load_voxels(np_array, pathfield)
print('Loaded voxels into scene.')
scene.finish()