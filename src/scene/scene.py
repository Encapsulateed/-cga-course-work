from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from .colors import *
from .common import Vector3D, Color


@dataclass
class Sphere:
    origin: Vector3D
    radius: float
    color: Color

    data_length: int = 7

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3] = self.radius
        data[4:7] = np.array(self.color)

        return data


@dataclass
class Light:
    origin: Vector3D

    data_length: int = 3

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)

        return data

@dataclass
class Rectangle:
    origin: Vector3D
    u_vect: Vector3D
    v_vect: Vector3D
    color: Color

    normal_orientation : np.float32
    
    data_length: int = 13

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3:6] = np.array(self.u_vect)
        data[6:9] = np.array(self.v_vect)
        data[9:12] = np.array(self.color)
        data[12] = np.array(self.normal_orientation)
        return data


@dataclass
class Paraboloid:
    origin: Vector3D
    a: np.float32
    b: np.float32
    color: Color

    
    data_length: int = 8

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3] = np.array(self.a)
        data[4] = np.array(self.b)
        data[5:8] = np.array(self.color)
        return data
@dataclass
class Plane:
    origin: Vector3D
    normal: Vector3D
    color: Color

    data_length: int = 9

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3:6] = np.array(self.normal) / np.linalg.norm(np.array(self.normal))
        data[6:9] = np.array(self.color)

        return data


class Scene:
    def __init__(self, lights: List[Light], spheres: List[Sphere], planes: List[Plane], rectangles : List[Rectangle],
                 paraboloids : List[Paraboloid]):
        self.lights = lights
        self.spheres = spheres
        self.planes = planes
        self.rectangles = rectangles
        self.paraboloids = paraboloids
        
    def get_spheres(self) -> np.ndarray:
        data = np.zeros((Sphere.data_length, len(self.spheres)), dtype=np.float32)

        for i, s in enumerate(self.spheres):
            data[:, i] = s.to_array()

        return data

    def get_reactangles(self) -> np.ndarray:
        data = np.zeros((Rectangle.data_length, len(self.rectangles)), dtype=np.float32)
        
        for i,r in enumerate(self.rectangles):
            data[:,i] = r.to_array() 
        return data
    
    def get_parabaloids(self) -> np.ndarray:
        data = np.zeros((Paraboloid.data_length, len(self.paraboloids)), dtype=np.float32)
        
        for i,p in enumerate(self.paraboloids):
            data[:,i] = p.to_array() 
        return data
    
    def get_planes(self) -> np.ndarray:
        data = np.zeros((Plane.data_length, len(self.planes)), dtype=np.float32)

        for i, p in enumerate(self.planes):
            data[:, i] = p.to_array()

        return data

    def get_lights(self) -> np.ndarray:
        data = np.zeros((Light.data_length, len(self.lights)), dtype=np.float32)

        for i, l in enumerate(self.lights):
            data[:, i] = l.to_array()

        return data

    def generate_scene(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray):
        return self.get_spheres(), self.get_lights(), self.get_planes(),self.get_reactangles(), self.get_parabaloids()

    @staticmethod
    def default_scene() -> Scene:

        lights = [
                  Light([-2, 0, 3.0]),Light([2, 0, 3.0])]

        spheres = [
                   Sphere([1, 1, 0.4], 0.4, BLUE),
     ]

        planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]
        
        rectangles = [Rectangle(origin=[2, 2.5, 2] , u_vect= [-1,4,0] , v_vect= [0,0,5],color=AQUA, normal_orientation=0),
                       Rectangle(origin=[-2.001, 2, 2] , u_vect= [-1,4,0] , v_vect= [0,0,5],color=GREEN, normal_orientation=0),
                       Rectangle(origin=[-2, 2, 2] , u_vect= [-1,4,0] , v_vect= [0,0,5],color=GREEN, normal_orientation=1)]
        
        paraboloids= []
        
        return Scene(lights, spheres, planes,rectangles, paraboloids)