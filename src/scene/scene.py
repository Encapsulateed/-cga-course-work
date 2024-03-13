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
        
        l = []
        for r in self.rectangles:
            dx = 0.001
            a = r.u_vect
            b = r.v_vect
            
            k = 1 
            
            if r.normal_orientation == 0:
                k=-1
            
            N = (k*(a[1]*b[2] - a[2]*b[1]), k*(a[2]*b[0] - a[0]*b[2]),k*(a[0]*b[1] - a[1]*b[0]))
            
            if N[0] >= 0:
                l.append(Rectangle(origin=[r.origin[0] - dx, r.origin[1],r.origin[2]] , u_vect= r.u_vect ,v_vect= r.v_vect,
                                    color=r.color, normal_orientation=(r.normal_orientation+1)%2))
        
        #self.rectangles+=l
        
        l = []
        for p in self.paraboloids:
            dz = 0.001
            if p.orientation == -1:
                dz*=-1
            l.append(Paraboloid(origin=[p.origin[0],p.origin[1],p.origin[2] + dz],color=p.color,a=p.a,b=p.b,n_orient= p.n_orient*-1,h=p.h-dz,
                                orientation=p.orientation))
        self.paraboloids +=l

        
     

        
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
                Light([0,0,0.2]), Light([0,0,3.8])]
#,Light([0,6,4]),Light([0,-6,4])    Light([-7,0,3]), 
        spheres = [
                   Sphere([0, 0, 1.5], 0.1,     RED)
                  #Sphere([-3, 0, 3], 0.2,     AQUA)
     ]

        planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]
        
        rectangles = [Rectangle(origin=[-5, 5, 0.01] , u_vect= [0,10,0] , v_vect= [-10,0,0],color=GREEN, normal_orientation=1),]
                      #Rectangle(origin=[5, 5, 5] , u_vect= [0,10,0] , v_vect= [0,0,5],color=AQUA, normal_orientation=-1)]
        #,Rectangle(origin=[-2.001, 2, 2] , u_vect= [-1,4,0] , v_vect= [0,0,5],color=GREEN, normal_orientation=0)
        paraboloids= [#Paraboloid(origin=[0,0,0],orientation= 1,a=1,b=1,color=YELLOW,h=1,n_orient=1),
                      Paraboloid(origin=[0,0,4],orientation=-1,a=1,b=1,color=RED,h=3,n_orient=1),
                     ]
        
        spheres = []
       # rectangles = []
        #paraboloids = []      

        #rectangles.append(Rectangle(origin=[-2.001, 2, 2] , u_vect= [-1,4,0] , v_vect= [0,0,5],color=GREEN, normal_orientation=0))
        return Scene(lights, spheres, planes,rectangles, paraboloids)