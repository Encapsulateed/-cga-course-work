from numba import cuda
from math import sqrt
from .common import normalize, dot, vector_difference


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin: tuple, ray_dir: tuple, sphere_origin: tuple, sphere_radius: float) -> float:


    R = normalize(ray_dir)

    L = vector_difference(sphere_origin, ray_origin)

    a = dot(R, R)
    b = 2 * dot(L, R)
    c = dot(L, L) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0.0:
        return -999.9
    else:
        numerator = -b - sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)

        numerator = -b + sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)
        else:
            return -999.0


@cuda.jit(device=True)
def intersect_ray_plane(ray_origin: tuple, ray_dir: tuple, plane_origin: tuple, plane_normal: tuple) -> float:

    EPS = 0.001

    N = plane_normal[0:3]

    denom = dot(ray_dir, plane_normal)

    if abs(denom) < EPS:
        return -999.9

    LP = vector_difference(ray_origin, plane_origin)

    nominator = dot(LP, N)

    dist = nominator / denom

    if dist > 0:
        return dist
    else:
        return -999.0