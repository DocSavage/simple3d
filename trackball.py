# Taichi implementation of a virtual trackball by William Katz
# from C code:
# 
#   Implemented by Gavin Bell, lots of ideas from Thant Tessman and
#   the August '88 issue of Siggraph's "Computer Graphics," pp. 121-129.
#   
#   Original code from:
#   David M. Ciemiewicz, Mark Grossman, Henry Moreton, and Paul Haeberli
#  
#   Much mucking with by:
#   Gavin Bell

import taichi as ti
import taichi.math as tm

# This size should really be based on the distance from the center of
# rotation to the point on the object underneath the mouse.  That
# point would then track the mouse as closely as possible.  This is a
# simple example, though, so that is left as an Exercise for the
# Programmer.
TRACKBALLSIZE  = 8.6 #4.3

# Pass the x and y coordinates of the last and current positions of
# the mouse, scaled so they are from (-1.0 ... 1.0).
#
# Then simulate a track-ball.  Project the points onto the virtual
# trackball, then figure out the axis of rotation, which is the cross
# product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
# Note:  This is a deformed trackball-- is a trackball in the center,
# but is deformed into a hyperbolic sheet of rotation away from the
# center.  This particular function was chosen after trying out
# several variations.
# 
# The resulting rotation is returned as a quaternion rotation in the
# first paramater.
@ti.func
def trackball(p1: tm.vec2, p2: tm.vec2) -> tm.mat4:
    # If "from" and "to" are the same, return the identity quaternion.
    q = tm.vec4(0.0, 0.0, 0.0, 1.0)
    if p1[0] != p2[0] or p1[1] != p2[1]:
        # First, figure out z-coordinates for projection of P1 and P2 to
        # deformed sphere
        p1_3d = tm.vec3(p1[0], p1[1], project_to_sphere(TRACKBALLSIZE, p1))
        p2_3d = tm.vec3(p2[0], p2[1], project_to_sphere(TRACKBALLSIZE, p2))

        # Now, we want the cross product of P1 and P2
        axis = tm.cross(p2_3d, p1_3d)

        # Figure out how much to rotate around that axis.
        delta = p2_3d - p1_3d
        t = tm.length(delta) / (2.0 * TRACKBALLSIZE)

        # Avoid problems with out-of-control values...
        if t > 1.0:
            t = 1.0
        if t < -1.0:
            t = -1.0
        phi = 2.0 * ti.asin(t)

        q = axis_to_quat(axis, phi)
    return build_rotmatrix(q)


# Given an axis and angle, compute quaternion.
@ti.func
def axis_to_quat(a: tm.vec3, phi: float) -> tm.vec4:
    tm.normalize(a)
    scaling = ti.sin(phi / 2.0)
    return tm.vec4(a[0] * scaling, a[1] * scaling, a[2] * scaling, ti.cos(phi / 2.0))


# Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
# if we are away from the center of the sphere.
@ti.func
def project_to_sphere(r: float, v: tm.vec2) -> ti.float32:
    d = v.norm()
    x = 0.0
    if d < r * 0.70710678118654752440: # Inside sphere
        x = ti.sqrt(r * r - d * d)
    else:                              # On hyperbola
        t = r / 1.41421356237309504880
        x = t * t / d
    return x

# Build a rotation matrix, given a quaternion rotation.
@ti.func
def build_rotmatrix(q: tm.vec4) -> tm.mat4:
    # q = tm.normalize(q)
    # m = tm.mat4(
    #     1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2],
    #     2.0 * q[0] * q[1] - 2.0 * q[3] * q[2],
    #     2.0 * q[0] * q[2] + 2.0 * q[3] * q[1],
    #     0.0,

    #     2.0 * q[0] * q[1] + 2.0 * q[3] * q[2],
    #     1.0 - 2.0 * q[0] * q[0] - 2.0 * q[2] * q[2],
    #     2.0 * q[1] * q[2] - 2.0 * q[3] * q[0],
    #     0.0,

    #     2.0 * q[0] * q[2] - 2.0 * q[3] * q[1],
    #     2.0 * q[1] * q[2] + 2.0 * q[3] * q[0],
    #     1.0 - 2.0 * q[0] * q[0] - 2.0 * q[1] * q[1],
    #     0.0,

    #     0.0,
    #     0.0,
    #     0.0,
    #     1.0
    # )
    m = tm.mat4(
        1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]),
        2.0 * (q[0] * q[1] + q[2] * q[3]),
        2.0 * (q[2] * q[0] - q[1] * q[3]),
        0.0,

        2.0 * (q[0] * q[1] - q[2] * q[3]),
        1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]),
        2.0 * (q[1] * q[2] + q[0] * q[3]),
        0.0,

        2.0 * (q[2] * q[0] + q[1] * q[3]),
        2.0 * (q[1] * q[2] - q[0] * q[3]),
        1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]),
        0.0,

        0.0,
        0.0,
        0.0,
        1.0
    )
    return m
    