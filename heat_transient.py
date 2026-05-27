import numpy as np
from globals import find_position
from node import Node
from tri3 import Tri3
from material import Material

class Heat_transient:

    def __init__(self, time_start: float, time_end, tot_increment: int, plot_interval: int):
        self.time_start: float = time_start
        self.time_end: float = time_end
        self.tot_increment: int = tot_increment
        self.plot_interval: int = plot_interval

    # ---------------------------------------------------------------------------
    def heat_solver(self, nodes, elements, materials, K, t_iniziale, callback=None):

        f, fix = self.assembly_loads(nodes)
        C = self.assembly_capacity(nodes, elements, materials)

        dt = (self.time_end - self.time_start) / self.tot_increment
        print(f"dt = {dt}")

        T = np.full(len(nodes), t_iniziale, dtype=float)

        free_index = np.where(np.array([n.fix[0] for n in nodes]) == 0)[0]
        fixed_index = np.where(np.array([n.fix[0] for n in nodes]) == 1)[0]
        T_fixed = np.array([n.dof[0] for n in nodes])[fixed_index]
        T[fixed_index] = T_fixed

        A = C / dt + K
        A_rid = A[np.ix_(free_index, free_index)]

        for step in range(self.tot_increment):

            t = self.time_start + (step + 1) * dt

            rhs = (C / dt) @ T
            rhs_free = rhs[free_index]
            rhs_free += f[free_index]
            rhs_free -= A[np.ix_(free_index, fixed_index)] @ T_fixed

            T_new_free = np.linalg.solve(A_rid, rhs_free)

            T[free_index] = T_new_free
            T[fixed_index] = T_fixed

            print(f"t={t:.3f}s — T min={T.min():.2f} max={T.max():.2f}")

            if callback:
                callback(T.copy(), t)

        return T
    # ---------------------------------------------------------------------------
    def assembly(self, nodes, elements, materials):
        tot_nodes = len(nodes)
        tot_elements = len(elements)
        if tot_elements <= 0 or tot_nodes <= 0:
            return None

        dim_dof = Node.NDOF
        K = np.zeros((tot_nodes * dim_dof, tot_nodes * dim_dof))

        for elem in elements:
            tot_el_nodes = len(elem.connectivity)
            pos, ns = [], []

            for id_node in elem.connectivity:
                p = find_position(id_node, nodes)
                if p >= 0:
                    pos.append(p)
                    ns.append(nodes[p])
                else:
                    print(f'ERROR: node {id_node} not found')
                    return None

            pos_mat = find_position(elem.id_mat, materials)
            if pos_mat is None:
                print(f'ERROR: material {elem.id_mat} not found')
                return None
            mat = materials[pos_mat]

            elK = elem.stiffness(ns, mat, elem.id)

            for node_row in range(tot_el_nodes):
                for node_col in range(tot_el_nodes):
                    for i in range(dim_dof):
                        row = dim_dof * pos[node_row] + i
                        row_el = dim_dof * node_row + i
                        for j in range(dim_dof):
                            col = dim_dof * pos[node_col] + j
                            col_el = dim_dof * node_col + j
                            K[row, col] += elK[row_el, col_el]
        return K
    # ---------------------------------------------------------------------------
    def assembly_loads(self, nodes):
        N = len(nodes)
        f = np.zeros(N)
        fix = np.zeros(N)

        for i, node in enumerate(nodes):
            if node.fix[0] == 1:
                fix[i] = 1
            else:
                f[i] = node.load  # solo Neumann
        return f, fix
    # ---------------------------------------------------------------------------
    def linear_solver(self, K, f, fix) -> np:
        delete_index = np.where(fix == 1)[0]
        free_index = np.where(fix == 0)[0]

        K_rid = K[np.ix_(free_index, free_index)]

        f_free = f[free_index]
        T_fixed = f[delete_index]
        K_cross = K[np.ix_(free_index, delete_index)]
        f_rid = f_free - K_cross @ T_fixed

        print("K_rid shape:", K_rid.shape)
        rank = np.linalg.matrix_rank(K_rid)
        if rank < K_rid.shape[0]:
            print(f"WARNING: K_rid singolare (rank={rank}/{K_rid.shape[0]})")
            exit()
        print(f"K_rid non singolare (rank={rank}/{K_rid.shape[0]})")

        temp_rid = np.linalg.solve(K_rid, f_rid)
        return (None, f_rid, temp_rid)
    # ---------------------------------------------------------------------------
    def assembly_capacity(self, nodes, elements, materials):
        tot_nodes = len(nodes)
        dim_dof = Node.NDOF
        C = np.zeros((tot_nodes * dim_dof, tot_nodes * dim_dof))

        for elem in elements:
            ns = [nodes[find_position(nid, nodes)] for nid in elem.connectivity]
            mat = materials[find_position(elem.id_mat, materials)]

            elC = elem.capacity(ns, mat)

            pos = [find_position(nid, nodes) for nid in elem.connectivity]
            for i, pi in enumerate(pos):
                for j, pj in enumerate(pos):
                    C[pi, pj] += elC[i, j]
        return C

    def heat_steady(self, nodes, elements, materials, K):

        f, fix = self.assembly_loads(nodes)

        free_index = np.where(fix == 0)[0]
        fixed_index = np.where(fix == 1)[0]

        T = np.zeros(len(nodes))
        T_fixed = np.array([n.dof[0] for n in nodes])[fixed_index]
        T[fixed_index] = T_fixed

        K_rid = K[np.ix_(free_index, free_index)]
        f_free = f[free_index]
        f_free -= K[np.ix_(free_index, fixed_index)] @ T_fixed

        rank = np.linalg.matrix_rank(K_rid)
        if rank < K_rid.shape[0]:
            print(f"WARNING: K_rid singolare (rank={rank}/{K_rid.shape[0]})")
            exit()
        print(f"K_rid non singolare (rank={rank}/{K_rid.shape[0]})")

        T[free_index] = np.linalg.solve(K_rid, f_free)

        print(f"Steady-state — T min={T.min():.2f} max={T.max():.2f}")
        return T