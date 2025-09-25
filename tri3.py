import numpy as np
import pygame as pg

from node import Node
from material import Material
from globals import find_position as fp

class Tri3:
    # ---------------------------------------------------------------------------
    def __init__(self, el_id: int, connectivity: list[int], id_mat: int, el_k) -> None:
        self.id = el_id
        self.connectivity = connectivity
        self.id_mat = id_mat
        self.el_k = el_k
    # ---------------------------------------------------------------------------
    def get_area(self, el_nodes: list[Node]) -> float:
        return 0.5 * np.abs(
        el_nodes[0].x[0] * (el_nodes[1].x[1] - el_nodes[2].x[1]) +
        el_nodes[1].x[0] * (el_nodes[2].x[1] - el_nodes[0].x[1]) +
        el_nodes[2].x[0] * (el_nodes[0].x[1] - el_nodes[1].x[1])
        )
    # ---------------------------------------------------------------------------
    def check_stretch(el_nodes: list[Node], area):
        base = max(
            np.sqrt((el_nodes[0].x[0] - el_nodes[1].x[0]) ** 2 + (el_nodes[0].x[1] - el_nodes[1].x[1]) ** 2),
            np.sqrt((el_nodes[1].x[0] - el_nodes[2].x[0]) ** 2 + (el_nodes[1].x[1] - el_nodes[2].x[1]) ** 2),
            np.sqrt((el_nodes[0].x[0] - el_nodes[2].x[0]) ** 2 + (el_nodes[0].x[1] - el_nodes[2].x[1]) ** 2)
        )
        height = area * 2 / base
        if base > 10 * height:
            print("WARNING: the element is stretched")
    # ---------------------------------------------------------------------------
    def get_B_matrix(self, el_nodes: list[Node], area) -> np:
        return np.array([
            [el_nodes[1].x[1] - el_nodes[2].x[1], el_nodes[2].x[1] - el_nodes[0].x[1],
            el_nodes[0].x[1] - el_nodes[1].x[1]],
            [el_nodes[2].x[0] - el_nodes[1].x[0], el_nodes[0].x[0] - el_nodes[2].x[0],
            el_nodes[1].x[0] - el_nodes[0].x[0]]
        ]) / (2 * area)
    # ---------------------------------------------------------------------------
    def stiffness(self, el_nodes: list[Node], mat: Material, elem_id: int) -> np:
        print(f"CALCOLO DI K - ELEMENTO {elem_id} -------------------------------------------------------------\n")
        area = self.get_area(el_nodes)
        print(f"Area dell'elemento: {area}")
        Tri3.check_stretch(el_nodes, area)
        B_matrix = self.get_B_matrix(el_nodes, area)
        stiffness_matrix = mat.cond * B_matrix.T @ B_matrix
        print(f"el_K: {stiffness_matrix}\n")
        return stiffness_matrix
    # ---------------------------------------------------------------------------
    def draw(self, screen, nodes, zoom, pan) -> None:
        vertices: list[pg.Vector2] = [
        pg.Vector2(*nodes[fp(self.connectivity[0], nodes)].x * zoom + pan),
        pg.Vector2(*nodes[fp(self.connectivity[1], nodes)].x * zoom + pan),
        pg.Vector2(*nodes[fp(self.connectivity[2], nodes)].x * zoom + pan),
        ]
        pg.draw.polygon(screen, (50,150,255), vertices, width=0)
        pg.draw.polygon(screen, (0,0,0), vertices, width=2)
    # ---------------------------------------------------------------------------