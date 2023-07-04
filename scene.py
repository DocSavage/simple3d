import time
import os
from datetime import datetime
import numpy as np
import taichi as ti
import taichi.math as tm

from trackball import trackball
from renderer import Renderer
from math_utils import np_normalize, np_rotate_matrix
import __main__


VOXEL_DX = 1 / 256
SCREEN_RES = (720, 720)
TARGET_FPS = 30
UP_DIR = (0, 1, 0)
HELP_MSG = '''
====================================================
Camera:
* Drag with your left mouse button to rotate
* Press W/A/S/D/Q/E to move
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

@ti.kernel
def calc_rotation_by_trackball(p0: tm.vec2, p1: tm.vec2) -> tm.mat4:
    return trackball(p0, p1)

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._lookat_pos = np.array((0.5, 0.5, 0.5))
        self._camera_pos = np.array((3.54648359, 3.17058634, 2.03934088))
        # self._camera_pos = np.array((3.1399074, 1.70, 2.72982264))
        # self._camera_pos = np.array((0.4, 0.5, 2.0))
        # self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None
        self._mouse_poll = 0

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self):
        res = self._update_by_keys()
        res = self._update_by_mouse() or res
        return res

    def _update_by_mouse(self):
        self._mouse_poll += 1
        if self._mouse_poll % 5 != 0:
            return False
        self._mouse_poll = 0
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        
        p0 = tm.vec2(*self._last_mouse_pos * 2.0 - 1)
        p1 = tm.vec2(*mouse_pos * 2.0 - 1)
        rotmatrix = calc_rotation_by_trackball(p1, p0)
        self._last_mouse_pos = mouse_pos

        # rotate the camera position while fixing the lookat position
        look_dir = self._camera_pos - self._lookat_pos
        look_dir_homo = np.array(list(look_dir) + [0.0])
        new_look_dir = np.matmul(rotmatrix, look_dir_homo)[:3]
        self._camera_pos = new_look_dir + self._lookat_pos


        # # Makes camera rotation feels right
        # dx, dy = self._last_mouse_pos - mouse_pos
        # self._last_mouse_pos = mouse_pos

        # out_dir = self._lookat_pos - self._camera_pos
        # leftdir = self._compute_left_dir(np_normalize(out_dir))

        # scale = 3
        # rotx = np_rotate_matrix(self._up, dx * scale)
        # roty = np_rotate_matrix(leftdir, dy * scale)

        # out_dir_homo = np.array(list(out_dir) + [0.0])
        # new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        # self._lookat_pos = self._camera_pos + new_out_dir

        # print(f"p0: {p0}, p1: {p1}")
        print(f"camera pos: {self._camera_pos}, lookat: {self._lookat_pos}, up: {self._up}")
        return True

    def _update_by_keys(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


class Scene:
    def __init__(self, voxel_edges=0.06, exposure=3, voxel_scope=512):
        ti.init(arch=ti.gpu) # modified for CUDA
        print(HELP_MSG)
        self.window = ti.ui.Window("Taichi Voxel Renderer",
                                   SCREEN_RES,
                                   vsync=True)
        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer = Renderer(dx=VOXEL_DX,
                                 image_res=SCREEN_RES,
                                 up=UP_DIR,
                                 voxel_edges=voxel_edges,
                                 voxel_scope=voxel_scope,
                                 exposure=exposure)

        self.renderer.set_camera_pos(*self.camera.position)
        if not os.path.exists('screenshot'):
            os.makedirs('screenshot')

    @staticmethod
    @ti.func
    def round_idx(idx_):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]),
             ti.round(idx[1]),
             ti.round(idx[2])]).cast(ti.i32)

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.renderer.set_voxel(self.round_idx(idx), mat, color)

    @ti.func
    def get_voxel(self, idx):
        mat, color = self.renderer.get_voxel(self.round_idx(idx))
        return mat, color

    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def set_directional_light(self, direction, direction_noise, color):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color

    def finish(self):
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        spp = 1
        while self.window.running:
            should_reset_framebuffer = False

            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            t = time.time()
            for _ in range(spp):
                self.renderer.accumulate()
            img = self.renderer.fetch_image()
            if self.window.is_pressed('p'):
                timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
                dirpath = os.getcwd()
                main_filename = os.path.split(__main__.__file__)[1]
                fname = os.path.join(dirpath, 'screenshot', f"{main_filename}-{timestamp}.jpg")
                ti.tools.image.imwrite(img, fname)
                print(f"Screenshot has been saved to {fname}")
            elif self.window.is_pressed('x'):     # Experimental -- not working yet 
                d = self.camera._camera_pos - self.camera._lookat_pos
                self.renderer.etch(self.camera._camera_pos, d)
                mouse_pos = np.array(self.window.get_cursor_pos())
                print(f"mouse pos: {mouse_pos}")
            elif self.window.is_pressed('c'):
                self.clip_z += 1
                if self.clip_z >= 256:
                    self.clip_z = 255
                if self.clip_z < 0:
                    self.clip_z = 0
                self.renderer.set_clip_z(self.clip_z)

            canvas.set_image(img)
            elapsed_time = time.time() - t
            if elapsed_time * TARGET_FPS > 1:
                spp = int(spp / (elapsed_time * TARGET_FPS) - 1)
                spp = max(spp, 1)
            else:
                spp += 1
            self.window.show()
