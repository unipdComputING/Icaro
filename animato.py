import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from read_gmsh import read_gmsh
from heat_transient import Heat_transient
from matplotlib.animation import FuncAnimation
# =========================================================
# MATERIALI
# =========================================================
class Material:
    def __init__(self, ID, dens, cond, cspec):
        self.id = ID
        self.dens = dens
        self.cond = cond
        self.cspec = cspec

materials = [Material(1, dens=200, cond=0.7, cspec=250000)]

# =========================================================
# MESH
# =========================================================
nodes, elements, materials, boundaries, sets, amps, loads = read_gmsh(
    path="t1.msh",
    materials=materials
)

print(f"Nodi:     {len(nodes)}")
print(f"Elementi: {len(elements)}")
print(f"Sets:     {[s.name for s in sets]}")

# =========================================================
# UTILITY
# =========================================================
def find_set(sets, keyword):
    for s in sets:
        if keyword.lower() in s.name.lower():
            return s
    return None

def find_node(nodes, nid):
    return next((n for n in nodes if n.id == nid), None)

top_set    = find_set(sets, "top")
bottom_set = find_set(sets, "bottom")
right_set = find_set(sets, "right")
# =========================================================
# DIRICHLET — temperature imposte
# =========================================================
T_bottom = 20.0
T_top    = 25.0
T_right    = 200

if bottom_set:
    for nid in bottom_set.get_list_of_entity():
        node = find_node(nodes, nid)
        if node:
            node.fix[0] = 1
            node.dof[0] = T_bottom

if top_set:
    for nid in top_set.get_list_of_entity():
        node = find_node(nodes, nid)
        if node:
            node.fix[0] = 1
            node.dof[0] = T_top

if right_set:
    for nid in right_set.get_list_of_entity():
        node = find_node(nodes, nid)
        if node:
            node.fix[0] = 1
            node.dof[0] = T_right

print(f"Dirichlet bottom: {T_bottom}°C — {len(bottom_set.get_list_of_entity())} nodi")
print(f"Dirichlet top:    {T_top}°C — {len(top_set.get_list_of_entity())} nodi")
print(f"Dirichlet right:    {T_right}°C — {len(right_set.get_list_of_entity())} nodi")

# =========================================================
# SOLVER
# =========================================================
ht_solver = Heat_transient(
    time_start=0.0,
    time_end=1000.0,
    tot_increment=10,
    plot_interval=10
)

print("\n===== ASSEMBLAGGIO =====")
K = ht_solver.assembly(nodes, elements, materials)
print("K shape:", K.shape)

# Coordinate e triangolazione PRIMA del solve
x = np.array([n.x[0] for n in nodes])
y = np.array([n.x[1] for n in nodes])
triangles = np.array([
    [e.connectivity[0]-1, e.connectivity[1]-1, e.connectivity[2]-1]
    for e in elements
])
triang = mtri.Triangulation(x, y, triangles)

# Setup plot

print("\n===== SOLUZIONE =====")

x = np.array([n.x[0] for n in nodes])
y = np.array([n.x[1] for n in nodes])
triangles = np.array([
    [e.connectivity[0]-1, e.connectivity[1]-1, e.connectivity[2]-1]
    for e in elements
])
triang = mtri.Triangulation(x, y, triangles)


# =========================================================
# STORICO TEMPERATURE
# =========================================================

T_history = []
t_history = []

def save_step(T, t):
    T_history.append(T.copy())
    t_history.append(t)

# =========================================================
# SOLVE
# =========================================================

temp_out = ht_solver.heat_solver(
    nodes,
    elements,
    materials,
    K,
    callback=save_step      # <-- IMPORTANTE
)

# =========================================================
# ANIMAZIONE
# =========================================================

fig, ax = plt.subplots(figsize=(4, 8), num='ICARO')

print("Frames salvati:", len(T_history))

if len(T_history) == 0:
    raise ValueError("T_history è vuoto: nessun dato da animare")

contour = None
cbar = None

def animate(i):

    global contour, cbar

    ax.clear()

    contour = ax.tricontourf(
        triang,
        T_history[i],
        levels=20,
        cmap='hot',
        vmin=20,
        vmax=200
    )

    ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f't = {t_history[i]:.2f} s')

    ax.set_aspect('equal')

    if cbar is None:
        cbar = fig.colorbar(
            contour,
            ax=ax,
            label='Temperatura [°C]'
        )



ani = FuncAnimation(
    fig,
    animate,
    frames=len(T_history),
    interval=100,
    repeat=True,
    blit=False
)

plt.tight_layout()
plt.show()