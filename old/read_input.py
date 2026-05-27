# read input file
import numpy as np

from node import Node
from tri3 import Tri3
from material import Material
from boundary import Boundary
from set1 import Set1
from amp import Amp

def read_input(
        path: str = None,
        nodes: list[Node] = None,
        elements: list[Tri3]= None,
        materials: list[Material]= None,
        boundaries: list[Boundary] = None,
        sets: list[Set1] = None,
        amps: list[Amp] = None,
        loads: list= None
) -> tuple[list[Node], list[Tri3], list[Material], list[Boundary], list[Set1], list[Amp], list]:    #da aggiungere in output la lista dei materiali

  if path is None:
    print('ERROR in read_input: file path not defined')
    return ([], [], [], [], [], [], [])

  try:
    reader = open( path, "r" )
    txt = reader.read()
    reader.close()
    txt = txt.split( '\n' )
  except:
    print( 'ERROR: file not found' )
    return ([], [], [], [], [],[],[])
  dim: int = len( txt )
  iRow: int = 0

  while iRow < dim:     # fin quando l'indice di riga arriva a dim imposto il ciclo
    stCommand = txt[iRow].split('#')[0] # find the comment line, cosi nell'input file posso mettere i commenti
    if len(stCommand) > 0: stCommand = stCommand.split()[0].lower()
    # lowercase conversion
    # stCommand = stCommand.lower()   # letta la stringa, la trasformo con carattere minuscolo
    if stCommand == 'nodes':      # ricerca del blocco comandi
      iRow, nodes = read_nodes(iRow, txt, nodes)
    elif stCommand == 'elements':
      iRow, elements = read_elements(iRow, txt, elements)
    elif stCommand == 'materials':
      iRow, materials = read_materials(iRow, txt, materials)
    elif stCommand == 'boundaries':
      iRow, boundaries = read_boundary( iRow, txt, boundaries )
    elif stCommand == 'set':
      iRow, sets = read_sets ( iRow, txt, sets)
      # pass è da aggiungere quando il blocco è vuoto
    elif stCommand == 'amp':
      iRow, amps = read_amps(iRow, txt, amps)
    elif stCommand == 'solver':
      iRow, loads = read_solver(iRow, txt)

    iRow = iRow +1



  link_nodes_boundaries(boundaries, sets,  nodes) #funzione per aggiornare le condizioni al contorno dei nodi,

  return (nodes, elements, materials, boundaries, sets, amps, loads)    # forse qua bisogna aggiungere anche materials
# -----------------------------------------------------------------------------
nodes = []
def read_nodes(iRow, txt, nodes: list[Node]) -> tuple[int, list[Node]]:
  dim: int = len(txt)
  iRow = iRow + 1
  stLine = txt[iRow].split('#')[0].lower()    # suddivisione in sottostrighe utilizzando gli apazi come separatori
  while stLine != 'end' and iRow < dim:
    st: list = []
    st = stLine.split()
    try:
      if len(st) != 0:
        ID: int = int(st[0])
        x: np = np.array([float(st[1]), float(st[2])])
        nodes.append(Node(ID, x))
    except:
      print('ERROR in read_input: node not correctly defined')

    iRow += 1
    stLine = txt[iRow].split('#')[0].lower()



  return (iRow, nodes)
# -----------------------------------------------------------------------------
def read_elements(iRow, txt, elements: list[Tri3]) -> tuple[int, list[Tri3]]:
  dim: int = len(txt)
  iRow = iRow + 1
  stLine = txt[iRow].split('#')[0].lower()
  while stLine != 'end' and iRow < dim:
    st: list = []
    st = stLine.split()
    try:
      if len(st) != 0:
        ID: int = int(st[0])
        connectivity: list = [int(st[1]), int(st[2]), int(st[3])]
        id_mat: int = int(st[4])
        elements.append(Tri3(ID, connectivity, id_mat, np.zeros((3,3))))
    except:
      print('ERROR in read_input: element not correctly defined')

    iRow += 1
    stLine = txt[iRow].split('#')[0].lower()

  return (iRow, elements)
# -----------------------------------------------------------------------------
def read_materials(iRow, txt, materials: list[Material]) -> tuple[int, list[Material]]:
  dim: int = len( txt )
  iRow = iRow + 1
  stLine = txt[iRow].split( '#' )[0].lower()
  while stLine != 'end' and iRow < dim:
    st: list = []
    st = stLine.split()
    try:
      if len( st ) != 0:
        ID: int = int( st[0] )
        dens: float = float ( st[1] )
        cond: float = float ( st[2] )
        cspec: float = float ( st[3] )
        materials.append(Material( ID, dens , cond , cspec) )
    except:
      print( 'ERROR in read_input: material not correctly defined' )

    iRow += 1
    stLine = txt[iRow].split( '#' )[0].lower()

  return (iRow, materials)
# -----------------------------------------------------------------------------
def read_boundary(iRow, txt, boundaries: list[Boundary]) -> tuple[int, list[Boundary]]:
  global isinstance
  check_node_id: int

  dim: int = len(txt)
  iRow = iRow + 1


  stLine = txt[iRow].split('#')[0].lower()
  while stLine != 'end' and iRow < dim:
    st: list = []
    st = stLine.split()
    if len(st) != 0:
      name: str = None
      fix: np = None
      val: float = None
      amp: str = None
      try:
        check_node_id = int(st[0])  # Tenta di convertire st[0] in un intero
        node_id = check_node_id  # Se la conversione ha avuto successo, assegna il valore a node_id
      except ValueError:
        node_id = None  # Se non è un numero, imposta node_id a None


      if node_id is None:
        name = st[0]
        fix = np.array([int(st[1])])  # Assegna fix come un array di interi
        val = float(st[2])  # Assegna val come un float
        amp = st[3]
        boundaries.append(Boundary(node_id, name, fix, val, amp))  # Aggiungi la boundary con name
      else:

        fix = np.array([int(st[1])])
        val = float(st[2])
        amp = st[3] if len(st) > 3 else None




        boundaries.append(Boundary(node_id, None, fix, val, amp))  # Aggiungi la boundary con node_id

    iRow += 1
    stLine = txt[iRow].split('#')[0].lower()

  return (iRow, boundaries)
# -----------------------------------------------------------------------------
def read_sets (iRow, txt, sets: list[Set1]) -> tuple[int, list[Set1]]:
  dim: int = len( txt )
  stLine = txt[iRow].split( '#' )[0].lower()
  st: list = []
  st = stLine.split()
  name: str = ""
  try:
    if len( st ) != 0:
      name: str = (st[1])
  except:
    print( 'ERROR in read_input: set1 not correctly defined' )

  iRow = iRow + 1
  stLine = txt[iRow].split( '#' )[0].lower()
  while stLine != 'end' and iRow < dim: #serve ciclo while?
    #st: list = []
    st = stLine.split()
    try:
      if len( st ) != 0:
        list_of_entity: list = [int( num ) for num in st]
        dtype: list= [1,2]
        sets.append( Set1( name, list_of_entity) )
    except Exception as e:
      print(f"ERROR in read_input: {e} (line: {stLine})")
    iRow += 1
    stLine = txt[iRow].split( '#' )[0].lower()

  return (iRow, sets)
# -----------------------------------------------------------------------------
def read_amps(iRow, txt,amps: list[Amp]) -> tuple[int, list[Amp]] :
  dim: int = len(txt)
  stLine = txt[iRow].split('#')[0].lower()
  st: list = []
  st = stLine.split()
  name: str = ''

  if len(st) != 0:
      name: str = None
      try:
        name = st[1]
      except:
        print( 'amp name not defined' )
  iRow = iRow + 1
  stLine = txt[iRow].split('#')[0].lower()
  time: list[float] = []
  intensity: list[float] = []
  amp=Amp(name=name)

  while stLine != 'end' and iRow < dim:

    st = stLine.split()

    if len(st) != 0:

      try:

        time.append(float(st[0]))
        intensity.append(float(st[1]))
        amp.time.append(float(st[0]))
        amp.intensity.append(float(st[1]))
        amp.data.set_data(float(st[0]), float(st[1]))
      except ValueError as e:
        print(f"Error parsing line {iRow}: {st} - {e}")





    iRow += 1

    stLine = txt[iRow].split('#')[0].lower()

  #amps.append(Amp(name, time, intensity))
  amps.append(amp)
  return (iRow, amps)
# -----------------------------------------------------------------------------
def link_nodes_boundaries(boundaries, sets,  nodes):


  for boundary in boundaries:

    flag = int(boundary.fix[0])  # La seconda colonna che indica se applicare il vincolo
    set_name = boundary.set
    print(sets)  # Aggiungi questo per vedere cosa c'è nella lista sets

    if boundary.set is not None:  # Se esiste un set associato


      # Trova il set corrispondente
      node_set = next((s for s in sets if s.name == set_name), None)

      if flag == 1:
        # Imposto 'fix' su tutti i nodi del set
        if node_set != None:
          for node_id in node_set.get_list_of_entity():
           node = next((n for n in nodes if n.id == node_id), None)
           if node:
             node.fix[0]=1
             print(f"Node {node.id} updated: {node.fix}")


    elif boundary.node_id is not None:  # Se c'è un ID nodo

      if flag == 1:
        # Imposto 'fix' sul nodo specificato
        node = next((n for n in nodes if n.id == boundary.node_id), None)
        if node:
          node.fix[0]=1
          print(f"Node {node.id} updated: {node.fix}")
# -------------------------------------------------------------------------------------------------------
def read_solver(iRow, txt) -> tuple[int, list]:
  loads = []
  dim: int = len(txt)
  iRow = iRow + 1
  stLine = txt[iRow].split('#')[0].lower()    # suddivisione in sottostrighe utilizzando gli apazi come separatori
  while stLine != 'end' and iRow < dim:
    st = stLine.split()
    try:
      if len(st) != 0:
        loads.append(float(st[0]))
        loads.append(float(st[1]))
        loads.append(int(st[2]))
        loads.append(int(st[3]))

    except Exception as e:
      print('ERROR in solver: solver not correctly defined')
      print(f'{e}')


    iRow += 1
    stLine = txt[iRow].split('#')[0].lower()



  return (iRow, loads)
# -----------------------------------------------------------------------------
