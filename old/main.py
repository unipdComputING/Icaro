import pygame

from read_input import read_input
from node import Node
from tri3 import Tri3
from material import Material
from boundary import Boundary
from set1 import Set1
from amp import Amp
from heat_transient import Heat_transient

if __name__ == '__main__':
   pygame.init()
   screen = pygame.display.set_mode((800, 800))
   clock = pygame.time.Clock()
   print("PROGRAM START\n")

   path = '.\\ICARO v0.1\\oneEI3.txt'
   nodes: list[Node] = []
   elements: list[Tri3] = []
   materials: list[Material] = []
   boundaries: list[Boundary] = []
   sets1: list[Set1] = []
   amps: list[Amp] = []
   (nodes, elements, materials, boundaries, sets1, amps, times) = read_input(path, nodes, elements, materials, boundaries, sets1, amps)
   ht_solver = Heat_transient(times[0],times[1],times[2],times[3])
   K = ht_solver.assembly(nodes, elements, materials)
   print(f'MATRICE STIFFNESS K -----------------------------\n\n')
   temp_out = ht_solver.heat_solver(nodes, elements, materials, K)
   print(temp_out)
   pygame.quit()