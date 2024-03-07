from numba import cuda
from .intersections import intersect_ray_sphere, intersect_ray_plane
from .common import get_sphere_color, get_plane_color, get_sphere_normal, get_plane_normal, get_vector_to_light, \
    get_reflection, dot, linear_comb


@cuda.jit(device=True)
def get_intersection(ray_origin: tuple, ray_dir: tuple, spheres, planes) -> (float, int, int):


    intersect_dist = 999.0
    obj_index = -999
    obj_type = 404

    for idx in range(spheres.shape[1]):
        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 0

    for idx in range(planes.shape[1]):
        dist = intersect_ray_plane(ray_origin, ray_dir, planes[0:3, idx], planes[3:6, idx])

        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 1

    return intersect_dist, obj_index, obj_type


@cuda.jit(func_or_sig=None, device=True)
def trace(ray_origin: tuple, ray_dir: tuple, spheres, lights, planes, ambient_int: float, lambert_int: float) -> (tuple, tuple, tuple):


    RGB = (0.0, 0.0, 0.0)

    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    if obj_type == 404:
        return RGB, (404., 404., 404.), (404, 404., 404.)

    P = linear_comb(ray_origin, ray_dir, 1.0, intersect_dist)

    if obj_type == 0:         

        RGB_obj = get_sphere_color(obj_index, spheres)
        N = get_sphere_normal(P, obj_index, spheres)

    elif obj_type == 1:     

        RGB_obj = get_plane_color(obj_index, planes)
        N = get_plane_normal(obj_index, planes)

    else:                  
        return (0., 0., 0.), (404., 404., 404.), (404, 404., 404.)

    RGB = linear_comb(RGB, RGB_obj, 1.0, ambient_int)


    BIAS = 0.0002
    P = linear_comb(P, N, 1.0, BIAS)

    for light_index in range(lights.shape[1]):

        L = get_vector_to_light(P, lights, light_index)

        _, _, obj_type = get_intersection(P, L, spheres, planes)

        if obj_type != 404:
            continue

        lambert_intensity = lambert_int * dot(L, N)

        if lambert_intensity > 0:
            RGB = linear_comb(RGB, RGB_obj, 1.0, lambert_intensity)


    R = get_reflection(ray_dir, N)

    P = linear_comb(P, R, 1.0, BIAS)

    return RGB, P, R


@cuda.jit(device=True)
def sample(ray_origin: tuple, ray_dir: tuple, spheres, lights, planes, ambient_int, lambert_int,
           reflection_int, refl_depth) -> (tuple, tuple, tuple):

    RGB, POINT, REFLECTION_DIR = trace(ray_origin, ray_dir, spheres, lights, planes, ambient_int, lambert_int)

    for i in range(refl_depth):
        if (POINT[0] == 404. and POINT[1] == 404. and POINT[2] == 404.) or \
                (REFLECTION_DIR[0] == 404. and REFLECTION_DIR[1] == 404. and REFLECTION_DIR[2] == 404.):
            continue

        RGB_refl, POINT, REFLECTION_DIR = trace(POINT, REFLECTION_DIR, spheres, lights, planes, ambient_int, lambert_int)

        RGB = linear_comb(RGB, RGB_refl, 1.0, reflection_int ** (i + 1))

    return RGB