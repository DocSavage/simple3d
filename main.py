import taichi as ti
import taichi.math as tm

import sys
import numpy as np

ti.init(arch=ti.gpu)

# Load 3d grayscale volume where each voxel is a single byte (uint8) in file.
cube_side = 512
slice_shape = (cube_side, cube_side)
cube_shape = (cube_side, cube_side, cube_side)
filename = f'data/hemibrain_subvol_{cube_side}x{cube_side}x{cube_side}.bin'
try:
    with open(filename, 'rb') as f:
        data = f.read()
    print(f'Loaded {len(data)} bytes from binary file: {filename}')
    np_array = np.frombuffer(data, dtype=np.uint8).reshape(*cube_shape)
except FileNotFoundError:
    sys.exit(f'File {filename} not found.  Exiting...')

voxels = ti.field(dtype=ti.u8, shape=cube_shape)
pixels = ti.field(dtype=ti.u8, shape=slice_shape)

@ti.kernel
def load_voxels(bytedata: ti.types.ndarray()):
    for i, j, k in voxels:  # Parallelized over all voxels
        voxels[i, j, k] = bytedata[i, j, k]

@ti.kernel
def paint_xy(k: int):
    for i, j in pixels:  # Parallelized over all pixels
       pixels[i, j] = voxels[i, j, k]

# Now start the GUI and load the data into 3d field
gui = ti.GUI("Grayscale", res=slice_shape)
load_voxels(np_array)

# Cycle through each XY slice of the volume and show it.
k = 0
while gui.running:
    paint_xy(k)
    gui.set_image(pixels)
    gui.show()
    k = (k + 1) % cube_side