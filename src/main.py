import numpy as np
from numba import cuda
from ray_tracing import render
from scene import Scene, Camera
from viewer import convert_array_to_image

def main():
    w, h = 2000, 2000
    CAMERA = (-5,2,3)
    amb, lamb, refl, refl_depth = 0.1, 0.55, 0.4, 10
    aliasing = True

    scene = Scene.default_scene()
    spheres_host, light_host, planes_host,r_h ,p_h = scene.generate_scene()

    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)
    rectangles = cuda.to_device(r_h)
    parabaloids = cuda.to_device(p_h)
    
    camera = Camera(resolution=(w, h), position=CAMERA, euler=[0, -30, -40])

    camera_origin = cuda.to_device(camera.position)
    camera_rotation = cuda.to_device(camera.rotation)
    pixel_loc = cuda.to_device(camera.generate_pixel_locations())

    result = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(result.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(result.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    render[blockspergrid, threadsperblock](pixel_loc, result, camera_origin, camera_rotation,
                                           spheres, lights, planes, amb, lamb, refl, refl_depth, aliasing,rectangles,parabaloids,CAMERA)
    
    print ('compile_end')
    import time
    st = time.time()

    render[blockspergrid, threadsperblock](pixel_loc, result, camera_origin, camera_rotation,
                                           spheres, lights, planes, amb, lamb, refl, refl_depth, aliasing,rectangles,parabaloids,CAMERA)
    result = result.copy_to_host()
    et = time.time()
    print(f"time: {1000 * (et - st):,.1f} ms")

    print ('render end')

    
   
    image = convert_array_to_image(result)
    image.save('../output/img.png')

    return 0


if __name__ == '__main__':
    main()