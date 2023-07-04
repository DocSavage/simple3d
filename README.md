Experiments with [Taichi Language](https://taichi-lang.org) for 3d EM visualization
and image processing.

Main routines for various apps are files with `app_` prefix.

The path tracing rendering implementation was taken from the excellent Taichi
Voxel Challenge repository. An attempt was made to use Bell's virtual trackball
algorithm instead of the original voxel challenge tracking system, and the
`look at` point is the center of the 3d data so rotations are about its center.

## Installation

1. Install taichi

```bash
pip install taichi
```

2. Download the grayscale data from Dropbox and store it into the `data` subdirectory:

256 x 256 x 256 voxel cube -- use preferentially since code assumes it.
https://www.dropbox.com/scl/fi/5jp36gat2ipf4g2ni7ig2/hemibrain_subvol_256x256x256.bin?rlkey=keey6wehk8qby75ht2blmcx0c&dl=0

512 x 512 x 512 voxel cube.
https://www.dropbox.com/scl/fi/ftefxipjajmnfo37glsa3/hemibrain_subvol_512x512x512.bin?dl=0&rlkey=ejgtgfc9ajexe26e5jc6wksw9

3. Run one of the main files which start with `app_` prefix:

```bash
python app_pathtrace.py
```