from numba import cuda
from math import sqrt
from scene.common import Vector3D
from scene.scene import *

import numpy as np
@cuda.jit(device=True)
def to_tuple3(array):
    return (array[0], array[1], array[2])


@cuda.jit(device=True)
def linear_comb(a, b, c1, c2):
    p0 = c1 * a[0] + c2 * b[0]
    p1 = c1 * a[1] + c2 * b[1]
    p2 = c1 * a[2] + c2 * b[2]
    return (p0, p1, p2)


@cuda.jit(device=True)
def vector_difference(fromm, to):
    f0, f1, f2 = fromm
    t0, t1, t2 = to
    return (t0 - f0, t1 - f1, t2 - f2)


@cuda.jit(device=True)
def normalize(vector):
    (X, Y, Z) = vector
    norm = sqrt(X * X + Y * Y + Z * Z)
    return (X / norm, Y / norm, Z / norm)


@cuda.jit(device=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def matmul(A, x):

    A0, A1, A2 = A
    b0 = dot(A0, x)
    b1 = dot(A1, x)
    b2 = dot(A2, x)
    return (b0, b1, b2)


@cuda.jit(device=True)
def clip_color(color):

    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def clip_color_vector(color3):
    (R, G, B) = color3
    return clip_color(R), clip_color(B), clip_color(G)


@cuda.jit(device=True)
def get_sphere_color(index, spheres):

    (R, G, B) = spheres[4:7, index]
    return (R, G, B)


@cuda.jit(device=True)
def get_plane_color(index, planes):

    (R, G, B) = planes[6:9, index]
    return (R, G, B)

@cuda.jit(device=True)
def get_rectangle_color(index, rectangles):
    (R, G, B) = rectangles[9:12, index]
    return (R, G, B)

@cuda.jit(device=True)
def get_parabaloid_color(index, parabaloids):
    (R, G, B) = parabaloids[5:8, index]
    return (R, G, B)

@cuda.jit(device=True)
def get_vector_to_light(P, lights, light_index):

    L = lights[0:3, light_index]
    P_L = vector_difference(P, L)
    return normalize(P_L)


@cuda.jit(device=True)
def get_sphere_normal(P, sphere_index, spheres):
    sphere_origin = spheres[0:3, sphere_index]
    N = vector_difference(sphere_origin, P)
    return normalize(N)


@cuda.jit(device=True)
def get_plane_normal(plane_index, planes):
    return normalize(planes[3:6, plane_index])

@cuda.jit(device=True)
def get_rect_normal(rec_idx,rectangles):
    # [u,v] = n
    # 
    N = normalize(cross_product(rectangles[3:6, rec_idx],rectangles[6:9, rec_idx]))
    if(rectangles[12, rec_idx] == 0):
        N = (-N[0],-N[1],-N[2])
    return N

@cuda.jit(device=True)
def get_parabaloid_normal(P,p_idx, parabaloids):
    a = parabaloids[3,p_idx]
    b = parabaloids[4,p_idx]
    orient = parabaloids[8,p_idx] 
    C = parabaloids[0:3,p_idx]
    
    n_orient = parabaloids[10,p_idx]
    
    k = n_orient
    N = (k*((2*P[0] - 2*C[0])/a**2), k*((2*P[1] - 2*C[1])/b**2), k*(-1 * orient))
    
    return normalize(N)
 
@cuda.jit(device=True)
def get_reflection(ray_dir, normal):
    k = dot(ray_dir, normal)
    R = linear_comb(ray_dir, normal, 1.0, -2.0 * k)
    return R

@cuda.jit(device=True)
def cross_product(a: Vector3D, b: Vector3D) -> Vector3D:
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],a[0]*b[1] - a[1]*b[0])

@cuda.jit(device=True)
def is_ligth_inside_parabaloid(p_origin: tuple, l_origin: tuple, a: float, b:float, p_orinet):
    x_p, y_p, z_p = p_origin
    x_l, y_l, z_l = l_origin
    

    return ((z_l > z_p and p_orinet == 1) or ( z_l < z_p and p_orinet == -1)) and \
            ((x_p**2 / a**2)+(y_p**2 / b**2)) >= (((x_l**2 / a**2)+(y_l**2 / b**2)))

    