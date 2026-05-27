import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from tkinter import *
from tkinter import messagebox
import threading
from read_gmsh import read_gmsh
from heat_transient import Heat_transient

# =========================================================
# MATERIALS
# =========================================================
class Material:
    def __init__(self, ID, dens, cond, cspec):
        self.id   = ID
        self.dens = dens
        self.cond = cond
        self.cspec = cspec

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

# =========================================================
# GUI
# =========================================================
class IcaroApp(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent, bg="#1a1a2e")
        self.parent = parent

        # stato
        self.T_history = []
        self.t_history = []
        self.nodes = None
        self.elements = None
        self.materials_list = None
        self.sets = None
        self.triang = None
        self.x = None
        self.y = None

        self._build_ui()

    # ----------------------------------------------------------
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

        body = Frame(self, bg="#1a1a2e")
        body.pack(fill=BOTH, expand=True, padx=20, pady=15)

        left_outer = Frame(body, bg="#1a1a2e", width=340)
        left_outer.pack(side=LEFT, fill=Y, padx=(0, 20))
        left_outer.pack_propagate(False)

        left_canvas = Canvas(left_outer, bg="#1a1a2e", highlightthickness=0)
        left_vsb = Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side=RIGHT, fill=Y)
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        col_left = Frame(left_canvas, bg="#1a1a2e")
        left_canvas.create_window((0, 0), window=col_left, anchor="nw")

        def _on_frame_configure(e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        col_left.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(e):
            left_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        col_right = Frame(body, bg="#1a1a2e")
        col_right.pack(side=LEFT, fill=BOTH, expand=True)

        self._section_material(col_left)
        self._section_bc(col_left)
        self._section_time(col_left)
        self._section_plot(col_left)
        self._section_buttons(col_left)
        self._section_log(col_right)

    # ----------------------------------------------------------
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
        frame = LabelFrame(
            parent, text=f"  {title}  ",
            font=("Courier New", 9, "bold"),
            fg="#e94560", bg="#16213e",
            relief=FLAT, bd=1,
            highlightbackground="#e94560",
            highlightthickness=1
        )
        frame.pack(fill=X, pady=6)
        return frame

    # ----------------------------------------------------------
    def _section_material(self, parent):
        card = self._card(parent, "MATERIAL")
        self.v_dens  = StringVar(value="2500")
        self.v_cond  = StringVar(value="1.37")
        self.v_cspec = StringVar(value="880")
        self._label_entry(card, 0, "Density [kg/m³]",  self.v_dens)
        self._label_entry(card, 1, "Cond. [J/(s*m*K)]", self.v_cond)
        self._label_entry(card, 2, "C. spec [J/(kg*K)]", self.v_cspec)

    def _section_bc(self, parent):
        card = self._card(parent, "TEMPERATURE BORDI [°C]")
        self.v_Tbottom = StringVar(value="50")
        self.v_Ttop    = StringVar(value="0")
        self.v_Tleft   = StringVar(value="0")
        self.v_Tright  = StringVar(value="50")
        self.v_Tinit   = StringVar(value="0")
        self._label_entry(card, 0, "T bottom", self.v_Tbottom)
        self._label_entry(card, 1, "T top",    self.v_Ttop)
        self._label_entry(card, 2, "T left",   self.v_Tleft)
        self._label_entry(card, 3, "T right",  self.v_Tright)
        self._label_entry(card, 4, "T iniziale", self.v_Tinit)

    def _section_time(self, parent):
        card = self._card(parent, "PARAMETRI TEMPORALI")
        self.v_tstart = StringVar(value="0.0")
        self.v_tend = StringVar(value="1e4")
        self.v_nstep = StringVar(value="100")

        # --- checkbox modalità ---
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
        self._label_entry(card, 3, "n incrementi", self.v_nstep)


        self._time_entries = card

    def _toggle_time_params(self):
        state = "disabled" if self.v_mode.get() == "steady" else "normal"
        for widget in self._time_entries.winfo_children():
            if isinstance(widget, Entry):
                widget.configure(state=state)

    def _section_plot(self, parent):
        card = self._card(parent, "PARAMETRI PLOT")
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

    # ----------------------------------------------------------
    def _log(self, msg):
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.update_idletasks()

    def _clear_log(self):
        self.log_text.delete("1.0", END)

    # ----------------------------------------------------------
    def _run_solve(self):
        params = {
            "dens": float(self.v_dens.get()),
            "cond": float(self.v_cond.get()),
            "cspec": float(self.v_cspec.get()),
            "T_bottom": float(self.v_Tbottom.get()),
            "T_top": float(self.v_Ttop.get()),
            "T_left": float(self.v_Tleft.get()),
            "T_right": float(self.v_Tright.get()),
            "T_init": float(self.v_Tinit.get()),
            "t_start": float(self.v_tstart.get()),
            "t_end": float(self.v_tend.get()),
            "n_step": int(self.v_nstep.get()),
            "mode": self.v_mode.get(),
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
            mode = params["mode"]

            mode = self.v_mode.get()
            self._log(f"Mode: {'TRANSIENT (Backward Euler)' if mode == 'transient' else 'STEADY STATE'}")

            materials = [Material(1, dens=dens, cond=cond, cspec=cspec)]

            self._log(f"Material: ρ={dens} | k={cond} | c={cspec}")
            self._log(f"BC: bottom={T_bottom} top={T_top} left={T_left} right={T_right}")
            if mode == "transient":
                self._log(f"Time: {t_start} → {t_end} s  |  steps={n_step}")

            # mesh
            nodes, elements, materials, boundaries, sets, amps, loads = read_gmsh(
                path="t1.msh",
                materials=materials
            )
            self._log(f"Nodes: {len(nodes)}  |  Elements: {len(elements)}")

            # sets
            top_set = find_set(sets, "top")
            bottom_set = find_set(sets, "bottom")
            right_set = find_set(sets, "right")
            left_set = find_set(sets, "left")

            def apply_bc(s, T_val):
                if s:
                    for nid in s.get_list_of_entity():
                        node = find_node(nodes, nid)
                        if node:
                            node.fix[0] = 1
                            node.dof[0] = T_val

            apply_bc(bottom_set, T_bottom)
            apply_bc(top_set, T_top)
            apply_bc(left_set, T_left)
            apply_bc(right_set, T_right)

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

            fixed_ids = [n.id - 1 for n in nodes if n.fix[0] == 1]

            if mode == "transient":
                T_vec = np.full(len(nodes), T_init)
                for node in nodes:
                    if node.fix[0] == 1:
                        T_vec[node.id - 1] = node.dof[0]

                f_check = K @ T_vec
                flux_in = np.sum(f_check[fixed_ids][f_check[fixed_ids] > 0])
                flux_out = np.sum(f_check[fixed_ids][f_check[fixed_ids] < 0])

                self._log(f"\n{'─' * 45}")
                self._log(f"  Thermic equilibrium (t=0)")
                self._log(f"{'─' * 45}")
                self._log(f"  flux in  (BC): {flux_in:+.4e} W")
                self._log(f"  flux out (BC): {flux_out:+.4e} W")
                self._log(f"{'─' * 45}")
                self._log("\nStart heat_solver...")

                ht.heat_solver(
                    nodes, elements, materials, K,
                    t_iniziale=T_init,
                    callback=save_step
                )

            else:  # steady
                self._log("\nStart heat_steady...")
                T_ss = ht.heat_steady(nodes, elements, materials, K)
                T_history.append(T_ss.copy())
                t_history.append(0.0)

            # ── SALVATAGGIO STATO ──────────────────────────────────
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

            T_final = T_history[-1]
            f_final = K @ T_final
            flux_in_f = np.sum(f_final[fixed_ids][f_final[fixed_ids] > 0])
            flux_out_f = np.sum(f_final[fixed_ids][f_final[fixed_ids] < 0])
            residual_f = np.sum(f_final)
            scarto = abs(flux_in_f + flux_out_f) / max(abs(flux_in_f), abs(flux_out_f))

            label_t = "Steady State" if mode == "steady" else f"t={t_end}s"
            self._log(f"\n{'─' * 45}")
            self._log(f"  Thermic equilibrium ({label_t})")
            self._log(f"{'─' * 45}")
            self._log(f"  flux in  (BC): {flux_in_f:+.4e} W")
            self._log(f"  flux out (BC): {flux_out_f:+.4e} W")
            self._log(f"  Residual  ΣKT: {residual_f:+.4e} W")
            self._log(f"  Equilibrium:   {'✓ OK' if abs(residual_f) < 1e-6 else '✗ WARN ' + f'{residual_f:.2e}'}")
            self._log(f"{'─' * 45}")

            if mode == "transient":
                if scarto < 1e-1:
                    self._log("  ✓ STATIONARY")
                else:
                    self._log(f"  ⚠ TRANSITORY PHASE NOT COMPLETED (relative error={scarto:.1%})")
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
            self._log(f"\n[ERROR] {e}")
            import traceback
            self._log(traceback.format_exc())

    # ----------------------------------------------------------
    def _run_graph(self):
        if not self.T_history:
            messagebox.showwarning("ATTENTION", "NEED TO SOLVE.")
            return
        self._graph()

    def _graph(self):
        try:
            T_history = self.T_history
            t_history = self.t_history
            triang    = self.triang
            x, y      = self.x, self.y

            SMOOTH      = self.v_smooth.get()
            SIGMA       = float(self.v_sigma.get())
            GRID_RES    = int(self.v_grid_res.get())

            levels_iso  = int(self.v_levels_iso.get())
            cmap        = self.v_cmap.get()
            interval    = int(self.v_interval.get())

            step = float(self.v_step_cont.get())

            T_min = min(T.min() for T in T_history)
            T_max = max(T.max() for T in T_history)


            start = np.floor(T_min / step) * step
            end = np.ceil(T_max / step) * step

            levels_contour = np.arange(start, end + step, step)

            xc = x.mean()
            yc = y.mean()
            points = list(zip(x, y))

            fig, ax = plt.subplots(figsize=(4, 8), num='ICARO_001')

            if SMOOTH:
                xi = np.linspace(x.min(), x.max(), GRID_RES)
                yi = np.linspace(y.min(), y.max(), GRID_RES)
                Xi, Yi = np.meshgrid(xi, yi)

                def get_field(i):
                    Zi = LinearNDInterpolator(points, T_history[i])(Xi, Yi)
                    return gaussian_filter(Zi, sigma=SIGMA)

                Zi0 = get_field(0)
                contour = ax.contourf(Xi, Yi, Zi0, levels=levels_contour,
                                      cmap=cmap, vmin=T_min, vmax=T_max)
            else:
                def get_field(i):
                    return T_history[i]

                contour = ax.tricontourf(triang, T_history[0],
                                         levels=levels_contour,
                                         cmap=cmap, vmin=T_min, vmax=T_max)

            ax.triplot(triang, color='black', linewidth=0.3, alpha=0.4)
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Temperature [°C]", fontsize=8)
            t0_s = int(round(t_history[0]))
            t0_h = t_history[0] / 3600
            title = ax.set_title(f"t = {t0_s} s  ({t0_h:.2f} h)")
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
                    ax.contourf(Xi, Yi, Zi, levels=levels_contour,
                                cmap=cmap, vmin=T_min, vmax=T_max)
                    iso = ax.contour(Xi, Yi, Zi, levels=levels_iso,
                                     colors='black', linewidths=0.8)
                else:
                    ax.tricontourf(triang, T_history[i], levels=levels_contour,
                                   cmap=cmap, vmin=T_min, vmax=T_max)
                    iso = ax.tricontour(triang, T_history[i], levels=levels_iso,
                                        colors='black', linewidths=0.8)

                ax.clabel(iso, fmt="%.0f°", fontsize=9, inline=True,
                          inline_spacing=2, colors='black')
                ax.triplot(triang, color='black', linewidth=0.2, alpha=0.8)

                ax.text(xc, y.min(), f"{self.T_bottom:.0f}°",
                        ha='center', va='bottom', fontsize=9, color='black')
                ax.text(xc, y.max(), f"{self.T_top:.0f}°",
                        ha='center', va='top', fontsize=9, color='black')
                ax.text(x.min(), yc, f"{self.T_left:.0f}°",
                        ha='left', va='center', fontsize=9, color='black')
                ax.text(x.max(), yc, f"{self.T_right:.0f}°",
                        ha='right', va='center', fontsize=9, color='black')

                title = ax.set_title(f"t = {t_s} s  ({t_h:.2f} h)")

            ani = FuncAnimation(fig, animate, frames=len(T_history),
                                interval=interval, blit=False, repeat=False)

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

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    root = Tk()
    root.title("ICARO — Heat Solver")
    root.geometry("900x850")
    root.configure(bg="#1a1a2e")

    app = IcaroApp(root)
    app.pack(fill=BOTH, expand=True)

    root.mainloop()
