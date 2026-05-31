import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from tkinter import *
from tkinter import messagebox
import threading
import os
from read_gmsh import read_gmsh
from heat_transient import Heat_transient
from material import Material


def find_set(sets, keyword):
    for s in sets:
        if keyword.lower() in s.name.lower():
            return s
    return None

def find_node(nodes, nid):
    return next((n for n in nodes if n.id == nid), None)

def generate_ca_mesh(filename, W, H, copriferro, diam, n_arm_side,
                     mesh_size_max=None, mesh_size_min=None):
    import gmsh

    r = diam / 2.0
    n = max(0, int(n_arm_side))
    xl = copriferro + r
    xr = W - copriferro - r
    yb = copriferro + r
    yt = H - copriferro - r
    positions = []
    for i in range(n):
        t = i / (n-1) if n > 1 else 0.5
        positions.append((xl + t*(xr-xl), yb))
        positions.append((xl + t*(xr-xl), yt))
    if n > 2:
        n_vert = n - 2
        for j in range(1, n_vert+1):
            t = j / (n_vert+1)
            yv = yb + t*(yt-yb)
            positions.append((xl, yv))
            positions.append((xr, yv))
    positions = list({(round(x,8), round(y,8)) for x,y in positions})

    gmsh.initialize()
    gmsh.model.add("sezione_CA")
    fac = gmsh.model.geo

    p1 = fac.addPoint(0,0,0)
    p2 = fac.addPoint(W,0,0)
    p3 = fac.addPoint(W,H,0)
    p4 = fac.addPoint(0,H,0)
    l1 = fac.addLine(p1,p2)
    l2 = fac.addLine(p2,p3)
    l3 = fac.addLine(p3,p4)
    l4 = fac.addLine(p4,p1)
    outer_loop = fac.addCurveLoop([l1, l2, l3, l4])

    circle_loops   = []
    steel_surfaces = []

    for xc, yc in positions:
        pts = [
            fac.addPoint(xc+r, yc, 0),
            fac.addPoint(xc, yc+r, 0),
            fac.addPoint(xc-r, yc, 0),
            fac.addPoint(xc, yc-r, 0)
        ]
        arcs = [
            fac.addCircleArc(pts[0], fac.addPoint(xc,yc,0), pts[1]),
            fac.addCircleArc(pts[1], fac.addPoint(xc,yc,0), pts[2]),
            fac.addCircleArc(pts[2], fac.addPoint(xc,yc,0), pts[3]),
            fac.addCircleArc(pts[3], fac.addPoint(xc,yc,0), pts[0])
        ]
        loop = fac.addCurveLoop(arcs)
        circle_loops.append(loop)
        steel_surfaces.append(fac.addPlaneSurface([loop]))

    hole_loops = [-cl for cl in circle_loops]
    concrete_surface = fac.addPlaneSurface([outer_loop] + hole_loops)

    fac.synchronize()

    gmsh.model.addPhysicalGroup(2, [concrete_surface], 1, "mat_1")
    gmsh.model.addPhysicalGroup(2, steel_surfaces,     2, "mat_2")
    gmsh.model.addPhysicalGroup(1, [l1], 11, "bottom")
    gmsh.model.addPhysicalGroup(1, [l2], 12, "right")
    gmsh.model.addPhysicalGroup(1, [l3], 13, "top")
    gmsh.model.addPhysicalGroup(1, [l4], 14, "left")

    if mesh_size_max is None:
        mesh_size_max = min(W, H) / 20
    if mesh_size_min is None:
        mesh_size_min = diam / 6
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    return positions

class IcaroApp(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent, bg="#1a1a2e")
        self.parent = parent


        self.T_history = []
        self.t_history = []
        self.nodes = None
        self.elements = None
        self.materials_list = None
        self.sets = None
        self.triang = None
        self.x = None
        self.y = None

        self.mesh_nodes = None
        self.mesh_elements = None
        self.mesh_sets = None
        self.selected_nodes = set()
        self.node_bcs = {}

        self._current_mesh_path = "t1.msh"
        self.v_mesh_path = StringVar(value="t1.msh")
        self._mesh_elem_mat = None

        self._box_start = None
        self._box_rect = None

        self._build_ui()

    def _build_ui(self):

        header = Frame(self, bg="#0f3460", pady=10)
        header.pack(fill=X)

        txt = Text(header, bg="#0f3460", fg="#e94560",
                   font=("Courier New", 12),
                   height=1, bd=0, highlightthickness=0,
                   state="normal", cursor="arrow")
        txt.pack()

        txt.tag_configure("bold", font=("Courier New", 12, "bold"), underline=True)
        txt.tag_configure("normal", font=("Courier New", 12))

        words = "Interactive Computational thermal Analysis for Reinforced COncrete structures".split()
        for word in words:
            for char in word:
                if char.isupper():
                    txt.insert("end", char, "bold")
                else:
                    txt.insert("end", char, "normal")
            txt.insert("end", " ")

        txt.configure(state="disabled")

        Label(
            header,
            text="FEM · Backward Euler · 2D",
            font=("Courier New", 9),
            fg="#a0a0c0", bg="#0f3460"
        ).pack()
        topbar = Frame(
            self,
            bg="#111122",
            height=50
        )

        topbar.pack(
            fill=X,
            side=TOP
        )

        topbar.pack_propagate(False)

        body = Frame(self, bg="#1a1a2e")
        body.pack(fill=BOTH, expand=True, padx=20, pady=15)

        left_outer = Frame(body, bg="#1a1a2e", width=540)
        left_outer.pack(side=LEFT, fill=Y, padx=(0, 20))
        left_outer.pack_propagate(False)

        left_canvas = Canvas(left_outer, bg="#1a1a2e", highlightthickness=0)
        left_vsb = Scrollbar(left_outer, orient="vertical", command=left_canvas.yview, width=20)
        left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side=RIGHT, fill=Y)
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        col_left = Frame(left_canvas, bg="#1a1a2e")
        left_canvas.create_window((0, 0), window=col_left, anchor="nw")

        def _on_frame_configure(e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        col_left.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(e):
            x, y = e.x_root, e.y_root
            lx = left_outer.winfo_rootx()
            ly = left_outer.winfo_rooty()
            lw = left_outer.winfo_width()
            lh = left_outer.winfo_height()
            if lx <= x <= lx + lw and ly <= y <= ly + lh:
                left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def _on_scroll_up(e):
            x, y = e.x_root, e.y_root
            lx = left_outer.winfo_rootx()
            ly = left_outer.winfo_rooty()
            lw = left_outer.winfo_width()
            lh = left_outer.winfo_height()
            if lx <= x <= lx + lw and ly <= y <= ly + lh:
                left_canvas.yview_scroll(-1, "units")

        def _on_scroll_down(e):
            x, y = e.x_root, e.y_root
            lx = left_outer.winfo_rootx()
            ly = left_outer.winfo_rooty()
            lw = left_outer.winfo_width()
            lh = left_outer.winfo_height()
            if lx <= x <= lx + lw and ly <= y <= ly + lh:
                left_canvas.yview_scroll(1, "units")

        self.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Mac
        self.bind_all("<Button-4>", _on_scroll_up)  # Linux scroll su
        self.bind_all("<Button-5>", _on_scroll_down)  # Linux scroll giù

        left_outer.bind("<Enter>", lambda e: left_outer.bind_all("<MouseWheel>", _on_mousewheel))
        left_outer.bind("<Leave>", lambda e: left_outer.unbind_all("<MouseWheel>"))

        self._section_buttons(topbar)
        self._section_material(col_left)
        self._section_bc(col_left)
        self._section_mesh_gen(col_left)
        self._section_time(col_left)
        self._section_node_bc(col_left)
        self._section_plot(col_left)
        self._section_buttons(topbar)
        col_right = PanedWindow(
            body,
            orient=VERTICAL,
            bg="#1a1a2e",
            sashwidth=8,
            sashrelief=RAISED,
            bd=0
        )

        col_right.pack(
            side=LEFT,
            fill=BOTH,
            expand=True
        )

        mesh_frame = Frame(
            col_right,
            bg="#1a1a2e"
        )


        log_frame = Frame(
            col_right,
            bg="#1a1a2e",
            height=220
        )

        col_right.add(mesh_frame, stretch="always")
        col_right.add(log_frame)

        self._build_mesh_panel(mesh_frame)
        self._section_log(log_frame)

    def _build_mesh_panel(self, parent):
        frame = LabelFrame(parent, text="  MESH  ",
                           font=("Courier New", 9, "bold"),
                           fg="#e94560", bg="#16213e",
                           relief=FLAT, bd=1,
                           highlightbackground="#e94560",
                           highlightthickness=1)
        frame.pack(fill=BOTH, expand=True, pady=(0, 8))

        # toolbar mesh
        tb = Frame(frame, bg="#16213e")
        tb.pack(fill=X, padx=6, pady=4)

        def btn(text, cmd, color="#0f3460"):
            return Button(tb, text=text, command=cmd,
                          font=("Courier New", 8, "bold"),
                          fg="white", bg=color,
                          activebackground=color,
                          relief=FLAT, bd=0,
                          padx=8, pady=5, cursor="hand2")

        btn("📂 LOAD MESH", self._load_mesh).pack(side=LEFT, padx=2)
        btn("🔄 CLEAR SEL", self._clear_selection, "#333355").pack(side=LEFT, padx=2)


        self.lbl_sel = Label(tb, text="Selected nodes: 0",
                             font=("Courier New", 8),
                             fg="#a0a0c0", bg="#16213e")
        self.lbl_sel.pack(side=LEFT, padx=12)

        self.lbl_mode = Label(tb, text="[click: select node  |  drag: box selection]",
                              font=("Courier New", 7),
                              fg="#606080", bg="#16213e")
        self.lbl_mode.pack(side=RIGHT, padx=6)

        self.fig_mesh, self.ax_mesh = plt.subplots(figsize=(8, 7))
        self.fig_mesh.patch.set_facecolor("#16213e")
        self.ax_mesh.set_facecolor("#0a0a1a")
        self.ax_mesh.set_title("Load mesh", color="#a0a0c0", fontsize=9)

        self.canvas_mesh = FigureCanvasTkAgg(self.fig_mesh, master=frame)
        self.canvas_mesh.draw()
        self.canvas_mesh.get_tk_widget().pack(fill=BOTH, expand=True, padx=4, pady=4)
        self._scatter_sel = None


        self.canvas_mesh.mpl_connect("button_press_event", self._on_mesh_press)
        self.canvas_mesh.mpl_connect("motion_notify_event", self._on_mesh_drag)
        self.canvas_mesh.mpl_connect("button_release_event", self._on_mesh_release)

    def _section_node_bc(self, parent):
        card = self._card(parent, "BC SELECTED NODES")

        # tipo BC
        type_frame = Frame(card, bg="#16213e")
        type_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        self.v_node_bc_tipo = StringVar(value="dirichlet")
        Radiobutton(type_frame, text="Dirichlet (T)",
                    variable=self.v_node_bc_tipo, value="dirichlet",
                    font=("Courier New", 9), fg="#c0c0d0", bg="#16213e",
                    selectcolor="#e94560", activebackground="#16213e").pack(side=LEFT, padx=(0, 10))
        Radiobutton(type_frame, text="Neumann (q)",
                    variable=self.v_node_bc_tipo, value="neumann",
                    font=("Courier New", 9), fg="#c0c0d0", bg="#16213e",
                    selectcolor="#e94560", activebackground="#16213e").pack(side=LEFT)
        Radiobutton(type_frame, text="Free",
                    variable=self.v_node_bc_tipo, value="free",
                    font=("Courier New", 9), fg="#c0c0d0", bg="#16213e",
                    selectcolor="#e94560", activebackground="#16213e").pack(side=LEFT)

        self.v_node_bc_val = StringVar(value="0.0")
        self._label_entry(card, 1, "Value", self.v_node_bc_val)

        btn_frame = Frame(card, bg="#16213e")
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=6)

        def btn(text, cmd, color):
            return Button(btn_frame, text=text, command=cmd,
                          font=("Courier New", 8, "bold"),
                          fg="white", bg=color,
                          activebackground=color,
                          relief=FLAT, bd=0,
                          padx=8, pady=5, cursor="hand2")

        btn("✓ APPLY", self._apply_node_bc, "#1a6b3c").pack(side=LEFT, padx=2)
        btn("✗ REMOVE", self._remove_node_bc, "#7a2020").pack(side=LEFT, padx=2)

        Label(card, text="BC assigned nodes:",
              font=("Courier New", 8, "bold"),
              fg="#e94560", bg="#16213e").grid(row=3, column=0, columnspan=2,
                                               sticky="w", padx=8, pady=(8, 2))

        self.node_bc_list = Text(card,
                                 font=("Courier New", 8),
                                 bg="#0a0a1a", fg="#00ff88",
                                 height=5, width=30,
                                 relief=FLAT, bd=2)
        self.node_bc_list.grid(row=4, column=0, columnspan=2,
                               sticky="ew", padx=8, pady=(0, 6))

    def _load_mesh(self):
        path = self.v_mesh_path.get()
        self._current_mesh_path = path
        self._load_mesh_from_file(path)

    def _load_mesh_from_file(self, path):
        try:
            materials = [
                Material(1, dens=float(self.v_dens.get()),
                            cond=float(self.v_cond.get()),
                            cspec=float(self.v_cspec.get())),
                Material(2, dens=float(self.v_dens2.get()),
                            cond=float(self.v_cond2.get()),
                            cspec=float(self.v_cspec2.get())),
            ]
            nodes, elements, materials_r, boundaries, sets, amps, loads = read_gmsh(
                path=path, materials=materials)

            self.mesh_nodes    = nodes
            self.mesh_elements = elements
            self.mesh_sets     = sets
            self._current_mesh_path = path
            self.selected_nodes.clear()
            self.node_bcs.clear()

            x = np.array([n.x[0] for n in nodes])
            y = np.array([n.x[1] for n in nodes])
            triangles = np.array([
                [e.connectivity[0]-1, e.connectivity[1]-1, e.connectivity[2]-1]
                for e in elements])
            triang = mtri.Triangulation(x, y, triangles)

            self._mesh_x        = x
            self._mesh_y        = y
            self._mesh_triang   = triang
            self._mesh_elem_mat = np.array([e.id_mat for e in elements])

            parent_frame = self.canvas_mesh.get_tk_widget().master
            self.canvas_mesh.get_tk_widget().destroy()
            plt.close(self.fig_mesh)

            self.fig_mesh, self.ax_mesh = plt.subplots(figsize=(5, 7))
            self.fig_mesh.patch.set_facecolor("#16213e")
            self.ax_mesh.set_facecolor("#0a0a1a")


            self.canvas_mesh = FigureCanvasTkAgg(self.fig_mesh, master=parent_frame)
            self.canvas_mesh.get_tk_widget().pack(fill=BOTH, expand=True, padx=4, pady=4)


            self.canvas_mesh.mpl_connect("button_press_event", self._on_mesh_press)
            self.canvas_mesh.mpl_connect("motion_notify_event", self._on_mesh_drag)
            self.canvas_mesh.mpl_connect("button_release_event", self._on_mesh_release)


            self._redraw_mesh()

            self._log(f"Loaded mesh: {len(nodes)} nodes, {len(elements)} elem  [{path}]")
        except Exception as e:
            self._log(f"[ERROR LOAD MESH] {e}")
            import traceback
            self._log(traceback.format_exc())

    def _redraw_mesh(self):
        ax = self.ax_mesh
        ax.clear()
        ax.set_facecolor("#0a0a1a")

        if not hasattr(self, '_mesh_triang') or self._mesh_triang is None:
            ax.set_title("Load mesh", color="#a0a0c0", fontsize=9)
            self.canvas_mesh.draw()
            return

        triang = self._mesh_triang
        x, y = self._mesh_x, self._mesh_y

        elem_mat = getattr(self, "_mesh_elem_mat", None)
        if elem_mat is not None and len(elem_mat) > 0:
            import matplotlib.tri as mtri2
            mask_conc  = (elem_mat == 1)
            mask_steel = (elem_mat == 2)
            if np.any(mask_conc):
                tr_c = mtri2.Triangulation(x, y, triang.triangles[mask_conc])
                ax.tripcolor(tr_c, np.ones(len(x)), cmap="Blues",
                             vmin=0, vmax=2, shading="flat", alpha=0.35, zorder=1)
            if np.any(mask_steel):
                tr_s = mtri2.Triangulation(x, y, triang.triangles[mask_steel])
                ax.tripcolor(tr_s, np.ones(len(x))*1.8, cmap="YlOrBr",
                             vmin=0, vmax=2, shading="flat", alpha=0.7, zorder=2)
        self.canvas_mesh.draw()

        ax.triplot(triang, color="#3a5a8a", linewidth=0.4, alpha=0.6, zorder=3)

        if self.node_bcs:
            dir_ids  = [nid-1 for nid, bc in self.node_bcs.items() if bc["tipo"] == "dirichlet"]
            neu_ids  = [nid-1 for nid, bc in self.node_bcs.items() if bc["tipo"] == "neumann"]
            free_ids = [nid-1 for nid, bc in self.node_bcs.items() if bc["tipo"] == "free"]
            handles  = []
            import matplotlib.patches as mpatches
            if dir_ids:
                ax.scatter(x[dir_ids], y[dir_ids], c="#e94560", s=18, zorder=5)
                handles.append(mpatches.Patch(color="#e94560", label="Dirichlet"))
            if neu_ids:
                ax.scatter(x[neu_ids], y[neu_ids], c="#f0a500", s=18, zorder=5)
                handles.append(mpatches.Patch(color="#f0a500", label="Neumann"))
            if free_ids:
                ax.scatter(x[free_ids], y[free_ids], c="#00ccff", s=18, zorder=5)
                handles.append(mpatches.Patch(color="#00ccff", label="Free"))
            if handles:
                ax.legend(handles=handles, fontsize=7,
                          facecolor="#16213e", labelcolor="white", loc="upper right")

        if self.selected_nodes:
            ids = list(self.selected_nodes)
            ax.scatter(x[ids], y[ids], c="#00ff88", s=25, zorder=6)

        n  = len(self.mesh_nodes)
        nb = len(self.node_bcs)
        ax.set_title(
            f"Nodes: {n}  |  Elem: {len(self.mesh_elements)}  |  BC: {nb}  |  Sel: {len(self.selected_nodes)}",
            color="#a0a0c0", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(colors="#606080", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#303050")

        self.canvas_mesh.draw_idle()
        self.lbl_sel.config(text=f"Node selected: {len(self.selected_nodes)}")

    def _find_nearest_node(self, xdata, ydata):
        if self.mesh_nodes is None:
            return None
        x, y = self._mesh_x, self._mesh_y
        dist = (x - xdata)**2 + (y - ydata)**2
        return int(np.argmin(dist))

    def _on_mesh_press(self, event):
        if event.inaxes != self.ax_mesh or self.mesh_nodes is None:
            return
        self._box_start = (event.xdata, event.ydata)
        self._box_rect  = None

    def _on_mesh_drag(self, event):
        if self._box_start is None or event.inaxes != self.ax_mesh:
            return
        x0, y0 = self._box_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return

        if self._box_rect:
            self._box_rect.remove()
        w, h = x1-x0, y1-y0
        self._box_rect = Rectangle((min(x0,x1), min(y0,y1)), abs(w), abs(h),
                                    linewidth=1, edgecolor="#00ff88",
                                    facecolor="#00ff8820", zorder=10)
        self.ax_mesh.add_patch(self._box_rect)
        self.canvas_mesh.draw_idle()

    def _on_mesh_release(self, event):
        if self._box_start is None or self.mesh_nodes is None:
            return
        x0, y0 = self._box_start
        self._box_start = None

        if event.xdata is None or event.ydata is None:
            if self._box_rect:
                self._box_rect.remove()
                self._box_rect = None
            self.canvas_mesh.draw_idle()
            return

        x1, y1 = event.xdata, event.ydata
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        tol = (self._mesh_x.max() - self._mesh_x.min()) * 0.01

        if dx < tol and dy < tol:

            idx = self._find_nearest_node(x0, y0)
            if idx is not None:
                if idx in self.selected_nodes:
                    self.selected_nodes.discard(idx)
                else:
                    self.selected_nodes.add(idx)
        else:
            xmin, xmax = min(x0,x1), max(x0,x1)
            ymin, ymax = min(y0,y1), max(y0,y1)
            x, y = self._mesh_x, self._mesh_y
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            for idx in np.where(mask)[0]:
                self.selected_nodes.add(int(idx))

        if self._box_rect:
            self._box_rect.remove()
            self._box_rect = None

        self._redraw_mesh()
        self.canvas_mesh.flush_events()

    def _clear_selection(self):
        self.selected_nodes.clear()
        if self.mesh_nodes:
            self._redraw_mesh()
        self.lbl_sel.config(text="Selected nodes: 0")

    def _apply_node_bc(self):
        if not self.selected_nodes:
            messagebox.showwarning("Attention!", "0 nodes selected.")
            return
        tipo = self.v_node_bc_tipo.get()
        try:
            val = float(self.v_node_bc_val.get())
        except ValueError:
            messagebox.showerror("Error", "Value not valid.")
            return

        for idx in self.selected_nodes:
            node_id = self.mesh_nodes[idx].id
            self.node_bcs[node_id] = {"tipo": tipo, "val": val}


        self._update_node_bc_list()
        self._redraw_mesh()
        self._log(f"BC {tipo} val={val} assigned to {len(self.selected_nodes)} nodes")

    def _remove_node_bc(self):
        if not self.selected_nodes:
            messagebox.showwarning("Attention!", "0 nodes selected.")
            return
        removed = 0
        for idx in self.selected_nodes:
            node_id = self.mesh_nodes[idx].id
            if node_id in self.node_bcs:
                del self.node_bcs[node_id]
                removed += 1
        self._update_node_bc_list()
        self._redraw_mesh()
        self._log(f"Removed BC from {removed} nodes")

    def _update_node_bc_list(self):
        self.node_bc_list.delete("1.0", END)
        for node_id, bc in sorted(self.node_bcs.items()):
            tipo = bc["tipo"]
            val = bc["val"]
            if tipo == "dirichlet":
                line = f"  node  {node_id:>5d}  T = {val:+.2f}\n"
            elif tipo == "neumann":
                line = f"  node {node_id:>5d}  q = {val:+.2f}\n"
            else:
                line = f"  node  {node_id:>5d}  FREE\n"
            self.node_bc_list.insert(END, line)

    def _label_entry(self, parent, row, label, var, width=12):
        Label(
            parent, text=label,
            font=("Courier New", 9),
            fg="#c0c0d0", bg="#16213e",
            anchor="w", width=18
        ).grid(row=row, column=0, sticky="w", padx=8, pady=3)
        Entry(
            parent, textvariable=var,
            width=width,
            font=("Courier New", 9),
            bg="#0f3460", fg="#e0e0f0",
            insertbackground="white",
            relief=FLAT, bd=4
        ).grid(row=row, column=1, sticky="w", padx=8, pady=3)

    def _card(self, parent, title):
        outer = Frame(parent, bg="#e94560", padx=1, pady=1)
        outer.pack(fill=X, pady=6, padx=4)

        frame = LabelFrame(
            outer, text=f"  {title}  ",
            font=("Courier New", 9, "bold"),
            fg="#e94560", bg="#16213e",
            relief=FLAT, bd=0,
            highlightthickness=0
        )
        frame.pack(fill=X)

        def on_focus_in(e):
            outer.config(bg="#ff6b6b", padx=2, pady=2)
            frame.config(bg="#1e2a4a", fg="#ff6b6b")
            _recolor(frame, "#1e2a4a")

        def on_focus_out(e):
            outer.config(bg="#e94560", padx=1, pady=1)
            frame.config(bg="#16213e", fg="#e94560")
            _recolor(frame, "#16213e")

        def _recolor(widget, color):
            for child in widget.winfo_children():
                try:
                    child.config(bg=color)
                except:
                    pass
                _recolor(child, color)

        def _bind_focus(widget):
            widget.bind("<FocusIn>", on_focus_in, add="+")
            widget.bind("<FocusOut>", on_focus_out, add="+")
            for child in widget.winfo_children():
                _bind_focus(child)

        frame.bind("<Map>", lambda e: _bind_focus(frame))

        return frame

    def _section_material(self, parent):
        card = self._card(parent, "MATERIALS")

        Label(card, text="▌ Concrete (mat_1)",
              font=("Courier New", 8, "bold"),
              fg="#e94560", bg="#16213e", anchor="w"
              ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0))

        self.v_dens = StringVar(value="2500")
        self.v_cond = StringVar(value="1.37")
        self.v_cspec = StringVar(value="880")
        self._label_entry(card, 1, "Density [kg/m³]", self.v_dens)
        self._label_entry(card, 2, "Cond. [J/(s·m·K)]", self.v_cond)
        self._label_entry(card, 3, "C. spec [J/(kg·K)]", self.v_cspec)

        Label(card, text="▌ Steel (mat_2)",
              font=("Courier New", 8, "bold"),
              fg="#f0a500", bg="#16213e", anchor="w"
              ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 0))

        self.v_dens2 = StringVar(value="7850")
        self.v_cond2 = StringVar(value="50.0")
        self.v_cspec2 = StringVar(value="490")
        self._label_entry(card, 5, "Density [kg/m³]", self.v_dens2)
        self._label_entry(card, 6, "Cond. [J/(s·m·K)]", self.v_cond2)
        self._label_entry(card, 7, "C. spec [J/(kg·K)]", self.v_cspec2)

    def _browse_mesh(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select mesh",
            filetypes=[("Gmsh mesh", "*.msh"), ("All files", "*.*")]
        )
        if path:
            self.v_mesh_path.set(path)
            self._current_mesh_path = path
            self._load_mesh_from_file(path)

    def _section_mesh_gen(self, parent):
        card = self._card(parent, "GENERATE MESH CA")

        self.v_mg_W      = StringVar(value="0.30")
        self.v_mg_H      = StringVar(value="0.40")
        self.v_mg_cop    = StringVar(value="0.025")
        self.v_mg_diam   = StringVar(value="0.016")
        self.v_mg_n      = StringVar(value="2")
        self.v_mg_lc_max = StringVar(value="auto")
        self.v_mg_lc_min = StringVar(value="auto")
        self.v_mg_out = StringVar(value="t1.msh")
        self._label_entry(card, 8, "File output", self.v_mg_out)

        Label(card, text="▌ Load mesh",
              font=("Courier New", 8, "bold"),
              fg="#e94560", bg="#16213e", anchor="w"
              ).grid(row=11, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 0))

        path_frame = Frame(card, bg="#16213e")
        path_frame.grid(row=12, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

        Entry(path_frame, textvariable=self.v_mesh_path,
              width=20, font=("Courier New", 9),
              bg="#0f3460", fg="#e0e0f0",
              insertbackground="white",
              relief=FLAT, bd=4
              ).pack(side=LEFT, padx=(0, 4))

        Button(path_frame, text="📂",
               command=self._browse_mesh,
               font=("Courier New", 9),
               fg="white", bg="#333355",
               activebackground="#333355",
               relief=FLAT, bd=0,
               padx=6, pady=4, cursor="hand2"
               ).pack(side=LEFT)

        self._label_entry(card, 0, "Larghezza W [m]",    self.v_mg_W)
        self._label_entry(card, 1, "Altezza H [m]",      self.v_mg_H)
        self._label_entry(card, 2, "Copriferro [m]",     self.v_mg_cop)
        self._label_entry(card, 3, "Ø armatura [m]",     self.v_mg_diam)

        Label(card,
              text="  n = armature per lato (min 2 = solo angoli)",
              font=("Courier New", 7), fg="#606080", bg="#16213e", anchor="w"
              ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8)

        self._label_entry(card, 5, "N arm. per lato",    self.v_mg_n)
        self._label_entry(card, 6, "lc max [m / auto]",  self.v_mg_lc_max)
        self._label_entry(card, 7, "lc min [m / auto]",  self.v_mg_lc_min)
        self._label_entry(card, 8, "File output",        self.v_mg_out)

        btn_frame = Frame(card, bg="#16213e")
        btn_frame.grid(row=9, column=0, columnspan=2, sticky="w", padx=8, pady=8)

        def make_btn(text, cmd, color):
            return Button(btn_frame, text=text, command=cmd,
                          font=("Courier New", 8, "bold"),
                          fg="white", bg=color,
                          activebackground=color,
                          relief=FLAT, bd=0,
                          padx=10, pady=6, cursor="hand2")

        make_btn("⚙ Generate & load", self._generate_and_load, "#1a6b3c").pack(side=LEFT, padx=2)
        make_btn("👁 Preview",       self._preview_section,   "#0f3460").pack(side=LEFT, padx=2)

        self.lbl_mesh_info = Label(card,
                                   text="Nessuna mesh generata",
                                   font=("Courier New", 7),
                                   fg="#606080", bg="#16213e", anchor="w")
        self.lbl_mesh_info.grid(row=10, column=0, columnspan=2,
                                sticky="w", padx=8, pady=(0, 6))

    def _parse_mesh_params(self):
        try:
            W    = float(self.v_mg_W.get())
            H    = float(self.v_mg_H.get())
            cop  = float(self.v_mg_cop.get())
            diam = float(self.v_mg_diam.get())
            n    = int(self.v_mg_n.get())
            out  = self.v_mg_out.get().strip()
        except ValueError as e:
            messagebox.showerror("Errore mesh parameter", str(e))
            return None

        r = diam / 2.0
        if cop + r >= W / 2 or cop + r >= H / 2:
            messagebox.showerror("Errore geometria",
                "Copriferro + raggio armatura troppo grandi rispetto alla sezione!")
            return None

        lc_max_str = self.v_mg_lc_max.get().strip()
        lc_min_str = self.v_mg_lc_min.get().strip()
        lc_max = None if lc_max_str.lower() == "auto" else float(lc_max_str)
        lc_min = None if lc_min_str.lower() == "auto" else float(lc_min_str)

        return dict(W=W, H=H, cop=cop, diam=diam, n=n,
                    lc_max=lc_max, lc_min=lc_min, out=out)

    def _generate_and_load(self):
        p = self._parse_mesh_params()
        if p is None:
            return
        self._log("=== GENERATE MESH CA ===")
        self._log(f"  W={p['W']}m  H={p['H']}m  cov={p['cop']}m  Ø={p['diam']}m  n/lato={p['n']}")
        self.lbl_mesh_info.config(text="⏳ Generating...", fg="#f0a500")
        self.update_idletasks()
        try:
            positions = generate_ca_mesh(
                filename      = p["out"],
                W             = p["W"],
                H             = p["H"],
                copriferro    = p["cop"],
                diam          = p["diam"],
                n_arm_side    = p["n"],
                mesh_size_max = p["lc_max"],
                mesh_size_min = p["lc_min"],
            )
            n_arm = len(positions)
            sz    = os.path.getsize(p["out"])
            self._log(f"  ✓ Mesh generata: {n_arm} armature → {p['out']}  ({sz//1024} KB)")
            self.lbl_mesh_info.config(
                text=f"✓ {p['out']}  |  {n_arm} arm  |  {sz//1024} KB", fg="#00ff88")

            self.ax_mesh.clear()
            self.ax_mesh.set_title("Load mesh", color="#a0a0c0", fontsize=9)
            self.canvas_mesh.draw()

            self._load_mesh_from_file(p["out"])



        except Exception as e:
            import traceback
            self._log(f"[ERROR MESH] {e}")
            self._log(traceback.format_exc())
            self.lbl_mesh_info.config(text=f"✗ Error: {e}", fg="#e94560")

    def _preview_section(self):
        plt.close(self.fig_mesh)
        p = self._parse_mesh_params()
        if p is None:
            return
        import matplotlib.patches as mpatches
        W, H, cop, diam, n = p["W"], p["H"], p["cop"], p["diam"], p["n"]
        r = diam / 2.0
        xl, xr = cop + r, W - cop - r
        yb, yt = cop + r, H - cop - r
        positions = []
        for i in range(max(2, n)):
            t = i / (max(2,n)-1)
            positions.append((xl + t*(xr-xl), yb))
            positions.append((xl + t*(xr-xl), yt))
        if n > 2:
            for j in range(1, n-1):
                t = j / (n-1)
                positions.append((xl, yb + t*(yt-yb)))
                positions.append((xr, yb + t*(yt-yb)))
        uniq = list({(round(x,8), round(y,8)) for x,y in positions})

        fig, ax = plt.subplots(figsize=(4, 5), num="ICARO_PREVIEW")
        ax.set_facecolor("#0a0a1a")
        fig.patch.set_facecolor("#16213e")
        ax.add_patch(mpatches.Rectangle((0,0), W, H,
            linewidth=2, edgecolor="#e94560", facecolor="#1a3a6a", zorder=1))
        for xc, yc in uniq:
            ax.add_patch(mpatches.Circle((xc, yc), r,
                linewidth=1.5, edgecolor="#f0a500", facecolor="#c07800", zorder=3))
        ax.set_xlim(-W*0.05, W*1.05)
        ax.set_ylim(-H*0.05, H*1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{W*100:.0f}×{H*100:.0f} cm  |  {len(uniq)} arm Ø{diam*100:.1f}cm",
                     color="#a0a0c0", fontsize=8)
        ax.tick_params(colors="#606080", labelsize=7)
        plt.tight_layout()
        plt.show()

    def _section_bc(self, parent):
        card = self._card(parent, "BOUNDARY CONDITIONS")

        for col, (txt, w) in enumerate([("Border", 8), ("Type", 12), ("Value", 10)]):
            Label(
                card, text=txt,
                font=("Courier New", 8, "bold"),
                fg="#e94560", bg="#16213e",
                width=w, anchor="center"
            ).grid(row=0, column=col, padx=4, pady=(4, 2))

        self.bc_vars = {}
        borders = ["bottom", "top", "left", "right"]

        for i, name in enumerate(borders):
            tipo_var = StringVar(value="dirichlet")
            val_var = StringVar(value="0.0")
            self.bc_vars[name] = {"tipo": tipo_var, "val": val_var}

            Label(
                card, text=name,
                font=("Courier New", 9),
                fg="#c0c0d0", bg="#16213e",
                width=8, anchor="w"
            ).grid(row=i + 1, column=0, padx=8, pady=2, sticky="w")

            rf = Frame(card, bg="#16213e")
            rf.grid(row=i + 1, column=1, padx=4, pady=2, sticky="w")

            for val, label in [("dirichlet", "T"), ("neumann", "q"), ("none", "free")]:
                Radiobutton(
                    rf, text=label,
                    variable=tipo_var, value=val,
                    font=("Courier New", 11, "bold"),
                    fg="#c0c0d0", bg="#16213e",
                    selectcolor="#000000",
                    activebackground="#16213e",
                    indicatoron=True,
                    cursor="hand2",
                    padx=6, pady=4
                ).pack(side=LEFT, padx=6)

            Entry(
                card, textvariable=val_var,
                width=10,
                font=("Courier New", 9),
                bg="#0f3460", fg="#e0e0f0",
                insertbackground="white",
                relief=FLAT, bd=4
            ).grid(row=i + 1, column=2, padx=4, pady=2)

        Label(
            card,
            text="  T [°C] Dirichlet  |  q [W/m²] Neumann (q>0 enter)  |  free",
            font=("Courier New", 7),
            fg="#707090", bg="#16213e",
            anchor="w"
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))


        self.v_Tinit = StringVar(value="0")
        self._label_entry(card, 6, "Initial T [°C]", self.v_Tinit)

    def _section_time(self, parent):
        card = self._card(parent, "TIME PARAMETERS")
        self.v_tstart = StringVar(value="0.0")
        self.v_tend = StringVar(value="1e4")
        self.v_nstep = StringVar(value="100")

        self.v_mode = StringVar(value="transient")

        mode_frame = Frame(card, bg="#16213e")
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        Radiobutton(
            mode_frame, text="Transient",
            variable=self.v_mode, value="transient",
            font=("Courier New", 9), fg="#c0c0d0", bg="#16213e",
            selectcolor="#0f3460", activebackground="#16213e",
            command=self._toggle_time_params
        ).pack(side=LEFT, padx=(0, 10))

        Radiobutton(
            mode_frame, text="Steady State",
            variable=self.v_mode, value="steady",
            font=("Courier New", 9), fg="#c0c0d0", bg="#16213e",
            selectcolor="#0f3460", activebackground="#16213e",
            command=self._toggle_time_params
        ).pack(side=LEFT)

        self._label_entry(card, 1, "t start [s]", self.v_tstart)
        self._label_entry(card, 2, "t end [s]", self.v_tend)
        self._label_entry(card, 3, "n increments", self.v_nstep)


        self._time_entries = card

    def _toggle_time_params(self):
        state = "disabled" if self.v_mode.get() == "steady" else "normal"
        for widget in self._time_entries.winfo_children():
            if isinstance(widget, Entry):
                widget.configure(state=state)

    def _section_plot(self, parent):
        card = self._card(parent, "PLOT PARAMETERS")
        self.v_smooth      = BooleanVar(value=True)
        self.v_sigma       = StringVar(value="10")
        self.v_grid_res    = StringVar(value="300")
        self.v_step_cont = StringVar(value="5")
        self._label_entry(card, 3, "Step contourf [°C]", self.v_step_cont)
        self.v_levels_iso  = StringVar(value="4")
        self.v_cmap        = StringVar(value="rainbow")
        self.v_interval    = StringVar(value="100")

        Checkbutton(
            card, text="Smooth (Gaussian filter)",
            variable=self.v_smooth,
            font=("Courier New", 9),
            fg="#c0c0d0", bg="#16213e",
            selectcolor="#0f3460",
            activebackground="#16213e"
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=3)

        self._label_entry(card, 1, "Sigma smooth",    self.v_sigma)
        self._label_entry(card, 2, "Grid res",        self.v_grid_res)
        self._label_entry(card, 3, "Step contourf [°C]", self.v_step_cont)
        self._label_entry(card, 4, "N° isolines", self.v_levels_iso)
        self._label_entry(card, 5, "Colormap",        self.v_cmap)
        self._label_entry(card, 6, "Interval anim ms",self.v_interval)

    def _section_buttons(self, parent):
        btn_frame = Frame(parent, bg="#1a1a2e")
        btn_frame.pack(fill=X, pady=10)

        row1 = Frame(btn_frame, bg="#1a1a2e")
        row1.pack(fill=X)
        row2 = Frame(btn_frame, bg="#1a1a2e")
        row2.pack(fill=X, pady=(4, 0))

        def btn(frame, text, cmd, color):
            return Button(
                frame, text=text, command=cmd,
                font=("Courier New", 9, "bold"),
                fg="white", bg=color,
                activebackground=color,
                relief=FLAT, bd=0,
                padx=8, pady=7,
                cursor="hand2"
            )

        btn(row1, "▶ SOLVE", self._run_solve, "#e94560").pack(side=LEFT, padx=2)
        btn(row1, "📈 GRAPH", self._run_graph, "#0f3460").pack(side=LEFT, padx=2)
        btn(row1, "💾 EXPORT", self._export, "#1a6b3c").pack(side=LEFT, padx=2)
        btn(row1, "🗑 CLEAR", self._clear_log, "#333355").pack(side=LEFT, padx=2)

    def _section_log(self, parent):
        Label(
            parent, text="LOG",
            font=("Courier New", 9, "bold"),
            fg="#e94560", bg="#1a1a2e"
        ).pack(anchor="w")

        self.log_text = Text(
            parent,
            font=("Courier New", 11),
            bg="#0a0a1a", fg="#00ff88",
            insertbackground="white",
            relief=FLAT, bd=4,
            wrap=WORD
        )
        self.log_text.pack(fill=BOTH, expand=True)

        sb = Scrollbar(parent, command=self.log_text.yview)
        sb.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=sb.set)

    def _log(self, msg):
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.update_idletasks()

    def _clear_log(self):
        self.log_text.delete("1.0", END)

    def _run_solve(self):
        params = {
            "dens":     float(self.v_dens.get()),
            "cond":     float(self.v_cond.get()),
            "cspec":    float(self.v_cspec.get()),
            "dens2":    float(self.v_dens2.get()),
            "cond2":    float(self.v_cond2.get()),
            "cspec2":   float(self.v_cspec2.get()),
            "T_bottom": float(self.bc_vars["bottom"]["val"].get()),
            "T_top":    float(self.bc_vars["top"]["val"].get()),
            "T_left":   float(self.bc_vars["left"]["val"].get()),
            "T_right":  float(self.bc_vars["right"]["val"].get()),
            "T_init":   float(self.v_Tinit.get()),
            "t_start":  float(self.v_tstart.get()),
            "t_end":    float(self.v_tend.get()),
            "n_step":   int(self.v_nstep.get()),
            "mode":     self.v_mode.get(),
        }
        thread = threading.Thread(target=self._solve, args=(params,), daemon=True)
        thread.start()

    def _solve(self,params):
        try:
            self._log("=" * 45)
            self._log("")
            self._log("  ██╗ ██████╗ █████╗ ██████╗  ██████╗ ")
            self._log("  ██║██╔════╝██╔══██╗██╔══██╗██╔═══██╗")
            self._log("  ██║██║     ███████║██████╔╝██║   ██║")
            self._log("  ██║██║     ██╔══██║██╔══██╗██║   ██║")
            self._log("  ██║╚██████╗██║  ██║██║  ██║╚██████╔╝")
            self._log("  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ")
            self._log("")
            self._log("        ICARO - FEMDev 2024-2025")
            self._log("   Transient Heat Transfer Solver (2D FEM)")
            self._log("")
            self._log("=" * 45)
            self._log("")
            self._log("  START")
            self._log("=" * 45)

            dens = params["dens"]
            cond = params["cond"]
            cspec = params["cspec"]
            T_bottom = params["T_bottom"]
            T_top = params["T_top"]
            T_left = params["T_left"]
            T_right = params["T_right"]
            T_init = params["T_init"]
            t_start = params["t_start"]
            t_end = params["t_end"]
            n_step = params["n_step"]

            mode = self.v_mode.get()
            self._log(f"Mode: {'TRANSIENT (Backward Euler)' if mode == 'transient' else 'STEADY STATE'}")

            materials = [
                Material(1, dens=dens, cond=cond, cspec=cspec),
                Material(2, dens=params["dens2"], cond=params["cond2"], cspec=params["cspec2"]),
            ]

            self._log(f"Material: ρ={dens} | k={cond} | c={cspec}")
            self._log(f"BC: bottom={T_bottom} top={T_top} left={T_left} right={T_right}")
            if mode == "transient":
                self._log(f"Time: {t_start} → {t_end} s  |  steps={n_step}")

            # mesh
            nodes, elements, materials, boundaries, sets, amps, loads = read_gmsh(
                path=self._current_mesh_path,
                materials=materials
            )
            self._log(f"Nodes: {len(nodes)}  |  Elements: {len(elements)}")


            def apply_bc(s, T_val):
                if s:
                    for nid in s.get_list_of_entity():
                        node = find_node(nodes, nid)
                        if node:
                            node.fix[0] = 1
                            node.dof[0] = T_val

            def apply_flux(nodes, edge_list, q_value):
                for (id_a, id_b) in edge_list:
                    node_a = find_node(nodes, id_a)
                    node_b = find_node(nodes, id_b)
                    if node_a is None or node_b is None:
                        continue
                    L = node_a.distance(node_b)
                    node_a.load += q_value * L / 2
                    node_b.load += q_value * L / 2

            set_map = {
                "bottom": find_set(sets, "bottom"),
                "top": find_set(sets, "top"),
                "left": find_set(sets, "left"),
                "right": find_set(sets, "right"),
            }

            for border, vars_ in self.bc_vars.items():
                tipo = vars_["tipo"].get()
                val = float(vars_["val"].get())
                s = set_map[border]
                if s is None:
                    self._log(f"  [{border}] set NOT FOUND")
                    continue


                if tipo == "dirichlet":
                    apply_bc(s, val)


                elif tipo == "neumann":
                    nids = s.get_list_of_entity()

                    border_nodes = [find_node(nodes, nid) for nid in nids]
                    border_nodes = [n for n in border_nodes if n is not None]

                    if border in ("bottom", "top"):
                        border_nodes.sort(key=lambda n: n.x[0])
                    else:
                        border_nodes.sort(key=lambda n: n.x[1])


                    edge_list = [(border_nodes[i].id, border_nodes[i + 1].id)
                                 for i in range(len(border_nodes) - 1)]
                    apply_flux(nodes, edge_list, val)


            for node_id, bc in self.node_bcs.items():
                node = find_node(nodes, node_id)
                if node is None:
                    continue
                tipo = bc["tipo"]
                val = bc["val"]
                if tipo == "dirichlet":
                    node.fix[0] = 1
                    node.dof[0] = val
                elif tipo == "neumann":
                    node.load += val
                elif tipo == "free":
                    node.fix[0] = 0
                    node.dof[0] = 0.0
                    node.load = 0.0

            # solver
            ht = Heat_transient(
                time_start=t_start,
                time_end=t_end,
                tot_increment=n_step,
                plot_interval=10
            )

            self._log("\nAssembly K...")

            K = ht.assembly(nodes, elements, materials)
            self._log(f"K shape: {K.shape}")

            x = np.array([n.x[0] for n in nodes])
            y = np.array([n.x[1] for n in nodes])
            triangles = np.array([
                [e.connectivity[0] - 1, e.connectivity[1] - 1, e.connectivity[2] - 1]
                for e in elements
            ])
            triang = mtri.Triangulation(x, y, triangles)

            T_history = []
            t_history = []

            def save_step(T, t):
                T_history.append(T.copy())
                t_history.append(t)

            self._log("\nSolution in progress...")


            if mode == "transient":
                T_vec = np.full(len(nodes), T_init)
                for node in nodes:
                    if node.fix[0] == 1:
                        T_vec[node.id - 1] = node.dof[0]

                self._log(f"\n{'─' * 45}")
                self._log("\nStart heat_solver...")

                ht.heat_transient_solver(
                    nodes, elements, materials, K,
                    t_iniziale=T_init,
                    callback=save_step
                )

            else:  # steady
                self._log("\nStart heat_steady...")
                T_ss = ht.heat_steady_solver(nodes, elements, materials, K)
                T_history.append(T_ss.copy())
                t_history.append(0.0)


            self.T_history = T_history
            self.t_history = t_history
            self.nodes = nodes
            self.elements = elements
            self.triang = triang
            self.x = x
            self.y = y
            self.T_bottom = T_bottom
            self.T_top = T_top
            self.T_left = T_left
            self.T_right = T_right

            if mode == "transient":
                dT = T_history[-1] - T_history[-2]
                err = np.linalg.norm(dT) / np.linalg.norm(T_history[-1])
                if err < 1e-3:
                    self._log("  ✓ STATIONARY")
                else:
                    self._log(f"  ⚠ TRANSITORY PHASE NOT COMPLETED (relative error={err:.2%})")
            self._log(f"{'─' * 45}\n")

            scheme = "Backward Euler (implicit)" if mode == "transient" else "Direct linear solver"
            eq = "rho*c*dT/dt - div(k*grad(T)) = q" if mode == "transient" else "div(k*grad(T)) = 0"
            self._log(f"FEM equations:")
            self._log(f"    {eq}")
            self._log(f"    Scheme: {scheme}")
            self._log(f"    Nodes:     {len(nodes)}")
            self._log(f"    Elements:  {len(elements)}")
            self._log(f"    DOF:       {K.shape[0]}")
            self._log(f"    K sym:     {np.allclose(K, K.T)}")
            self._log(
                f"    K diag. dom.: {all(abs(K[i, i]) >= np.sum(abs(K[i, :])) - abs(K[i, i]) for i in range(K.shape[0]))}")
            self._log(f"    max|K_ij| = {np.max(np.abs(K)):.3e}")

            self._log(f"\nFrames: {len(T_history)}")
            self._log("✓ SOLUTION COMPLETED")
            self._log(f"{'─' * 45}\n")


        except Exception as e:

            self._log(f"\n[ERROR] {type(e).__name__}: {e}")

            import traceback

            self._log(traceback.format_exc())

    def _run_graph(self):
        if not self.T_history:
            messagebox.showwarning("ATTENTION", "NEED TO SOLVE.")
            return
        self._graph()

    def _graph(self):
        plt.close('all')
        try:
            T_history = self.T_history
            t_history = self.t_history
            triang = self.triang

            x, y = self.x, self.y

            SMOOTH = self.v_smooth.get()
            SIGMA = float(self.v_sigma.get())
            GRID_RES = int(self.v_grid_res.get())

            levels_iso = int(self.v_levels_iso.get())
            cmap = self.v_cmap.get()
            interval = int(self.v_interval.get())

            step = float(self.v_step_cont.get())

            T_min = min(T.min() for T in T_history)
            T_max = max(T.max() for T in T_history)

            start = np.floor(T_min / step) * step
            end = np.ceil(T_max / step) * step

            levels_contour = np.arange(start, end + step, step)

            xc = x.mean()
            yc = y.mean()
            points = list(zip(x, y))


            from collections import defaultdict

            elem_mat = np.array([e.id_mat for e in self.elements])

            triangles = np.array([
                [e.connectivity[0] - 1,
                 e.connectivity[1] - 1,
                 e.connectivity[2] - 1]
                for e in self.elements
            ])

            steel_tris = triangles[elem_mat == 2]

            edge_count = defaultdict(int)

            def add_edge(a, b):
                if a > b:
                    a, b = b, a
                edge_count[(a, b)] += 1

            for tri in steel_tris:
                add_edge(tri[0], tri[1])
                add_edge(tri[1], tri[2])
                add_edge(tri[2], tri[0])


            def draw_steel(ax, clear_previous=True):

                if clear_previous:
                    for line in ax.lines[:]:
                        if hasattr(line, '_is_steel') and line._is_steel:
                            line.remove()

                elem_mat = np.array([e.id_mat for e in self.elements])
                triangles = np.array([
                    [e.connectivity[0] - 1, e.connectivity[1] - 1, e.connectivity[2] - 1]
                    for e in self.elements
                ])
                steel_tris = triangles[elem_mat == 2]
                if len(steel_tris) == 0:
                    return

                steel_triang = mtri.Triangulation(self.x, self.y, steel_tris)


                lines = ax.triplot(steel_triang, color='black', linewidth=1.2, zorder=15)
                for l in lines:
                    l._is_steel = True


            fig, ax = plt.subplots(figsize=(8, 8), num='ICARO_001')

            if SMOOTH:
                xi = np.linspace(x.min(), x.max(), GRID_RES)
                yi = np.linspace(y.min(), y.max(), GRID_RES)
                Xi, Yi = np.meshgrid(xi, yi)

                def get_field(i):
                    Zi = LinearNDInterpolator(points, T_history[i])(Xi, Yi)
                    return gaussian_filter(Zi, sigma=SIGMA)

                Zi0 = get_field(0)
                contour = ax.contourf(
                    Xi, Yi, Zi0,
                    levels=levels_contour,
                    cmap=cmap,
                    vmin=T_min, vmax=T_max
                )
            else:
                def get_field(i):
                    return T_history[i]

                contour = ax.tricontourf(
                    triang,
                    T_history[0],
                    levels=levels_contour,
                    cmap=cmap,
                    vmin=T_min, vmax=T_max
                )

            draw_steel(ax)
            ax.triplot(triang, color='black', linewidth=0.3, alpha=0.4)

            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Temperature [°C]", fontsize=8)

            t0_s = int(round(t_history[0]))
            t0_h = t_history[0] / 3600
            ax.set_title(f"t = {t0_s} s  ({t0_h:.2f} h)")

            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            def animate(i):
                t_s = int(round(t_history[i]))
                t_h = t_history[i] / 3600

                for c in ax.collections:
                    c.remove()
                for t in ax.texts:
                    t.remove()

                if SMOOTH:
                    Zi = get_field(i)
                    ax.contourf(
                        Xi, Yi, Zi,
                        levels=levels_contour,
                        cmap=cmap,
                        vmin=T_min, vmax=T_max
                    )

                    iso = ax.contour(
                        Xi, Yi, Zi,
                        levels=levels_iso,
                        colors='black',
                        linewidths=0.8
                    )
                else:
                    ax.tricontourf(
                        triang,
                        T_history[i],
                        levels=levels_contour,
                        cmap=cmap,
                        vmin=T_min, vmax=T_max
                    )

                    iso = ax.tricontour(
                        triang,
                        T_history[i],
                        levels=levels_iso,
                        colors='black',
                        linewidths=0.8
                    )

                ax.clabel(
                    iso,
                    fmt="%.0f°",
                    fontsize=9,
                    inline=True,
                    inline_spacing=2,
                    colors='black'
                )
                draw_steel(ax, clear_previous=True)
                ax.triplot(triang, color='black', linewidth=0.2)


                ax.text(xc, y.min(), f"{self.T_bottom:.0f}°",
                        ha='center', va='bottom', fontsize=9, color='black')

                ax.text(xc, y.max(), f"{self.T_top:.0f}°",
                        ha='center', va='top', fontsize=9, color='black')

                ax.text(x.min(), yc, f"{self.T_left:.0f}°",
                        ha='left', va='center', fontsize=9, color='black')

                ax.text(x.max(), yc, f"{self.T_right:.0f}°",
                        ha='right', va='center', fontsize=9, color='black')

                ax.set_title(f"t = {t_s} s  ({t_h:.2f} h)")

            ani = FuncAnimation(
                fig,
                animate,
                frames=len(T_history),
                interval=interval,
                blit=False,
                repeat=False
            )

            plt.tight_layout()
            plt.show()
            plt.pause(0.01)

        except Exception as e:
            self._log(f"\n[ERRORE GRAPH] {e}")
            import traceback
            self._log(traceback.format_exc())

    def _export(self):
        if not self.T_history:
            messagebox.showwarning("Attention", " SOLVE.")
            return
        try:
            import meshio
            points = np.column_stack([self.x, self.y, np.zeros(len(self.x))])
            cells = [("triangle", np.array([
                [e.connectivity[0]-1, e.connectivity[1]-1, e.connectivity[2]-1]
                for e in self.elements
            ]))]
            point_data = {f"T_t{i}": T for i, T in enumerate(self.T_history)}
            mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
            mesh.write("icaro_output.vtk")
            self._log("✓ Exported: icaro_output.vtk")
        except Exception as e:
            self._log(f"[ERROR EXPORT] {e}")

if __name__ == "__main__":
    root = Tk()
    root.title("ICARO — Heat Solver")
    #root.attributes("-zoomed", True) #linux
    root.geometry("1920x1080") #windows
    root.configure(bg="#1a1a2e")

    app = IcaroApp(root)
    app.pack(fill=BOTH, expand=True)

    root.mainloop()
