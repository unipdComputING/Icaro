import numpy as np
import pygame

class Node:
  DIMSPACE: int = 2
  NDOF: int = 1
  radius = 5
  # ---------------------------------------------------------------------------
  def __init__(self, id: int = 0, position: np = None) -> None:
    self.id = id
    if position is None:
        self.x: np = np.zeros(self.DIMSPACE)
    else:
        self.x: np = position
    self.fix: np = np.zeros(self.NDOF, dtype='int')
    self.dof: np = np.zeros(self.NDOF, dtype='float')
    self.load: float = 0.0   # <-- aggiungi solo questa riga
  # ---------------------------------------------------------------------------
  def distance(self, node: 'Node') -> float:
    d: np = node.x - self.x
    return np.sqrt(np.dot(d, d))
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def draw(self, screen, zoom, pan) -> None:
    x_screen: pygame.Vector2 = pygame.Vector2(self.x[0] * zoom + pan[0], self.x[1] * zoom + pan[1])
    pygame.draw.circle(screen, (255,0,255), x_screen, self.radius, width=0)
    pygame.draw.circle(screen, (0,0,0), x_screen, self.radius, width=2)
  # ---------------------------------------------------------------------------
  # Creazione del dizionario
