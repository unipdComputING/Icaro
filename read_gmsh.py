# read_gmsh.py
import numpy as np
import meshio

from node import Node
from tri3 import Tri3
from material import Material
from boundary import Boundary
from set1 import Set1
from amp import Amp


def read_gmsh(
        path: str = None,
        materials: list = None,
        boundaries: list = None,
        sets: list = None,
        amps: list = None,
        loads: list = None
) -> tuple:

    if path is None:
        print('ERROR in read_gmsh: file path not defined')
        return ([], [], [], [], [], [], [])

    if materials is None: materials = []
    if boundaries is None: boundaries = []
    if sets is None: sets = []
    if amps is None: amps = []
    if loads is None: loads = []

    try:
        mesh = meshio.read(path)
    except Exception as e:
        print(f'ERROR in read_gmsh: cannot read file - {e}')
        return ([], [], [], [], [], [], [])

    # =========================================================
    # 1. NODI
    # Mappa gmsh_index (0-based) → node.id (1-based sequenziale)
    # =========================================================
    nodes = []
    gmsh_to_nodeid = {}

    for i, point in enumerate(mesh.points):
        node_id = i + 1
        gmsh_to_nodeid[i] = node_id
        x = np.array([float(point[0]), float(point[1])])
        nodes.append(Node(node_id, x))

    # =========================================================
    # 2. MAPPA tag gmsh → nome physical group
    # =========================================================
    tag_to_name = {}
    if mesh.field_data:
        for name, data in mesh.field_data.items():
            tag_to_name[int(data[0])] = name

    # =========================================================
    # 3. TAGS PER CELL BLOCK
    # =========================================================
    cell_tags_list = []
    if 'gmsh:physical' in mesh.cell_data:
        cell_tags_list = mesh.cell_data['gmsh:physical']
    else:
        cell_tags_list = [None] * len(mesh.cells)

    # =========================================================
    # 4. ELEMENTI E BOUNDARY NODES
    # =========================================================
    elements = []
    elem_id = 1
    boundary_nodes = {}  # nome → set di node_id (1-based)

    for block_idx, cell_block in enumerate(mesh.cells):
        etype = cell_block.type
        cells = cell_block.data

        tags = cell_tags_list[block_idx] if block_idx < len(cell_tags_list) else None

        for i, conn in enumerate(cells):
            # Converti indici gmsh 0-based → node.id 1-based tramite mappa
            node_ids = [gmsh_to_nodeid[int(n)] for n in conn]

            tag = int(tags[i]) if tags is not None else -1
            group_name = tag_to_name.get(tag, f"group_{tag}")

            if etype in ('triangle', 'triangle3'):
                mat_id = _mat_id_from_name(group_name, materials)
                coords = np.zeros((3, 2))
                for j, nid in enumerate(node_ids):
                    node = _find_node(nodes, nid)
                    if node is not None:
                        coords[j] = node.x
                elements.append(Tri3(elem_id, node_ids, mat_id, coords))
                elem_id += 1

            elif etype in ('line', 'line2', 'vertex'):
                if group_name not in boundary_nodes:
                    boundary_nodes[group_name] = set()
                boundary_nodes[group_name].update(node_ids)

    # =========================================================
    # 5. SET1
    # =========================================================
    for name, node_set in boundary_nodes.items():
        sets.append(Set1(name, sorted(list(node_set))))

    # =========================================================
    # 6. BOUNDARY (placeholder, fix=0)
    # =========================================================
    for s in sets:
        boundaries.append(Boundary(
            node_id=None,
            set=s.name,
            fix=np.array([0]),
            val=0.0,
            amp=None
        ))

    return (nodes, elements, materials, boundaries, sets, amps, loads)


# =========================================================
# HELPERS
# =========================================================
def _find_node(nodes, nid):
    return next((n for n in nodes if n.id == nid), None)


def _mat_id_from_name(name: str, materials: list) -> int:
    for mat in materials:
        if str(mat.id) in name:
            return mat.id
    return materials[0].id if materials else 1