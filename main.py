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

materials = [Material(1, dens=200, cond=0.7, cspec=25000000)]

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

left_set = find_set(sets, "left")
# =========================================================
# DIRICHLET — temperature imposte
# =========================================================
T_bottom = 50.0
T_top    = 0.0
T_right    = 50
T_left   = 50

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

if left_set:
    for nid in left_set.get_list_of_entity():
        node = find_node(nodes, nid)
        if node:
            node.fix[0] = 1
            node.dof[0] = T_left


print(f"Dirichlet bottom: {T_bottom}°C — {len(bottom_set.get_list_of_entity())} nodi")
print(f"Dirichlet top:    {T_top}°C — {len(top_set.get_list_of_entity())} nodi")
print(f"Dirichlet right:    {T_right}°C — {len(right_set.get_list_of_entity())} nodi")

# =========================================================
# SOLVER
# =========================================================
# SOLVER TRANSIENTE
# =========================================================

# =========================================================
# SOLVER TRANSIENTE
# =========================================================

ht_solver = Heat_transient(
    time_start=0.0,
    time_end=50.0,
    tot_increment=100,
    plot_interval=10
)

print("\n===== ASSEMBLAGGIO =====")

K = ht_solver.assembly(nodes, elements, materials)

print("K shape:", K.shape)

# =========================================================
# TRIANGOLAZIONE
# =========================================================

x = np.array([n.x[0] for n in nodes])
y = np.array([n.x[1] for n in nodes])

triangles = np.array([
    [
        e.connectivity[0]-1,
        e.connectivity[1]-1,
        e.connectivity[2]-1
    ]
    for e in elements
])

triang = mtri.Triangulation(x, y, triangles)

# =========================================================
# STORICO SOLUZIONI
# =========================================================

T_history = []
t_history = []

def save_step(T, t):

    T_history.append(T.copy())
    t_history.append(t)

# =========================================================
# SOLUZIONE
# =========================================================

print("\n===== SOLUZIONE =====")

temp_out = ht_solver.heat_solver(
    nodes,
    elements,
    materials,
    K,
    t_iniziale=20.0,
    callback=save_step
)

print("Frames salvati:", len(T_history))

if len(T_history) == 0:
    raise ValueError("T_history vuoto")
# =========================================================
# CONFIGURAZIONE PLOT
# =========================================================
SMOOTH = False
SMOOTH_SIGMA = 10    # solo se SMOOTH=True: 1-2 leggero, 5+ forte
GRID_RES = 300      # risoluzione griglia interpolazione

# =========================================================
# FIGURA
# =========================================================
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

fig, ax = plt.subplots(figsize=(4, 8), num='ICARO')

T_min = min(T.min() for T in T_history)
T_max = max(T.max() for T in T_history)

levels_contour = np.arange(np.ceil(T_min / 5) * 5, np.floor(T_max / 5) * 5 + 5, 5)
levels_label = 4
xc = x.mean()
yc = y.mean()
points = list(zip(x, y))

if SMOOTH:
    xi = np.linspace(x.min(), x.max(), GRID_RES)
    yi = np.linspace(y.min(), y.max(), GRID_RES)
    Xi, Yi = np.meshgrid(xi, yi)

    def get_field(i):
        Zi = LinearNDInterpolator(points, T_history[i])(Xi, Yi)
        return gaussian_filter(Zi, sigma=SMOOTH_SIGMA)

    # primo frame
    Zi0 = get_field(0)
    contour = ax.contourf(Xi, Yi, Zi0, levels=levels_contour, cmap='rainbow', vmin=T_min, vmax=T_max)
else:
    def get_field(i):
        return T_history[i]

    # primo frame
    contour = ax.tricontourf(triang, T_history[0], levels=levels_contour, cmap='rainbow', vmin=T_min, vmax=T_max)

ax.triplot(triang, color='black', linewidth=0.3, alpha=0.4)
cbar = fig.colorbar(contour, ax=ax)
title = ax.set_title(f"t = {t_history[0]:.1f} s")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

def animate(i):
    for c in ax.collections:
        c.remove()
    for t in ax.texts:
        t.remove()

    if SMOOTH:
        Zi = get_field(i)
        ax.contourf(Xi, Yi, Zi, levels=levels_contour, cmap='rainbow', vmin=T_min, vmax=T_max)
        iso = ax.contour(Xi, Yi, Zi, levels=levels_label, colors='black', linewidths=0.8)
    else:
        ax.tricontourf(triang, T_history[i], levels=levels_contour, cmap='rainbow', vmin=T_min, vmax=T_max)
        iso = ax.tricontour(triang, T_history[i], levels=levels_label, colors='black', linewidths=0.8)

    ax.clabel(iso, fmt="%.0f°", fontsize=10, inline=True, inline_spacing=2, colors='black')
    ax.triplot(triang, color='black', linewidth=0.2, alpha=0.8)

    ax.text(xc, y.min(), f"{T_bottom:.0f}°", ha='center', va='bottom', fontsize=9, color='black')
    ax.text(xc, y.max(), f"{T_top:.0f}°",    ha='center', va='top',    fontsize=9, color='black')
    ax.text(x.min(), yc, f"{T_left:.0f}°",   ha='left',   va='center', fontsize=9, color='black')
    ax.text(x.max(), yc, f"{T_right:.0f}°",  ha='right',  va='center', fontsize=9, color='black')

    title.set_text(f"t = {t_history[i]:.1f} s")

ani = FuncAnimation(fig, animate, frames=len(T_history), interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()