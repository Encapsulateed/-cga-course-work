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
    orientation : np.float32
    h :np.float32
    n_orient: np.float32
    
    data_length: int = 11

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3] = np.array(self.a)
        data[4] = np.array(self.b)
        data[5:8] = np.array(self.color)
        data[8] = np.array(self.orientation)
        data[9] = np.array(self.h)
        data[10] = np.array(self.n_orient)

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
            Light([0,0,4]), ]
        spheres = [
                  Sphere(origin=[-0.5,-0.5,1.1],radius=0.1,color=RED),
                   Sphere(origin=[0,1,0.4],radius=0.4,color=GREY),
                   Sphere(origin=[0,1,1.1],radius=0.3,color=YELLOW),
                   Sphere(origin=[0,1,1.6],radius=0.2,color=MAGENTA),
                    Sphere(origin=[-2.5,0,1.4],radius=0.1,color=SILVER),
                 
                
     ]

        spheres = [ Sphere(origin=[0,0,1.1],radius=0.1,color=SILVER),
                   Sphere(origin=[-1,-1,1.1],radius=0.1,color=YELLOW),
                    Sphere(origin=[-1,0,1.1],radius=0.1,color=AQUA),
                        Sphere(origin=[0,-1,1.1],radius=0.1,color=BLUE)
                    ]
        planes = [Plane([0, 0, 0], [0, 0, 1], GREY)]
                 # Plane([0, 0, 10], [0, 0, -1], GREY)]
        
        rectangles = [
                    #верх                   
                     Rectangle(origin=[0, 0, 1] , u_vect= [1,0,0] , v_vect= [0,1,0],color=GREEN, normal_orientation=1),
                    #зад   
                    Rectangle(origin=[0, 0, 1] , u_vect= [0,1,0] , v_vect= [0,0,2],color=GREEN, normal_orientation=1),
                      #передняя грань #
                       Rectangle(origin=[-1, 0, 1] , u_vect= [0,1,0] , v_vect= [0,0,2],color=GREEN, normal_orientation=1),
                      #правый бок #
                       Rectangle(origin=[0, -1, 1] , u_vect= [1,0,0] , v_vect= [0,0,1],color=GREEN, normal_orientation=1),
                        #левый бок #
                       Rectangle(origin=[0,  0, 1] , u_vect= [1,0,0] , v_vect= [0,0,1],color=GREEN, normal_orientation=1),
                       # -----------------------------
                        Rectangle(origin=[2, 1, 2] , u_vect= [3,4,0] , v_vect= [0,0,3],color=BLUE, normal_orientation=1)
                        ]
        paraboloids= [
                          Paraboloid(origin=[-2,-0.9,0],orientation=1,a=1,b=1,color=BLUE,h=1 ,n_orient=1)
       ]
        
      #  spheres = []
       # paraboloids = []
        #rectangles = []

        return Scene(lights, spheres, planes,rectangles, paraboloids)