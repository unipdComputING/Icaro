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
    def heat_solver(self, nodes: list[Node], elements: list[Tri3], materials: list[Material], K: np) -> np:
        dtime = (self.time_end - self.time_start) / self.tot_increment + 1
        dplot = (self.time_end - self.time_start) / self.plot_interval + 1
        temp: np = []
        temp_out: np = []
        f, fix = self.assembly_loads(nodes)  # assemblaggio dei carichi e vettore fix
        print(f'carichi: {f}\n')
        K_inv, f_rid = self.linear_solver(K, f, fix)
        time: float = 0.0
        plot: float = 0.0
        for inc in range(self.tot_increment):
            temp: np = np.dot(K_inv, f_rid)    # risoluzione del sistema lineare a = K^-1 * f
            if time == plot:
                temp_out = temp[inc]
                plot += dplot
            time += dtime

        return (temp_out)  # output

    # ---------------------------------------------------------------------------
    def assembly(self, nodes: list[Node], elements: list[Tri3], materials: list[Material]) -> np:
        tot_elements: int = len(elements)
        tot_nodes: int = len(nodes)
        if tot_elements <= 0 or tot_nodes <= 0:
            return None

        dim_dof: int = Node.NDOF
        dim_problem: int = tot_nodes * dim_dof

        K: np = np.zeros((dim_problem, dim_problem))

        cont_test: float = 1.0
        for elem in elements:
            tot_el_nodes: int = len(elem.connectivity)
            pos: list[int] = []
            ns: list[Node] = []
            for id_node in elem.connectivity:
                pos.append(find_position(id_node, nodes))
                if pos[-1] >= 0:
                    ns.append(nodes[pos[-1]])
                else:
                    print('ERROR in assembly: node %i not found' % (id_node))
                    return None
            pos_mat = find_position(elem.id_mat, materials)
            if pos_mat is None:
                print('ERROR in assembly: material %i not found' % (find_position))
                return None
            mat = materials[pos_mat]
            # ...............................insert the correct capacity matrix
            # elK: np = cont_test * np.ones((tot_el_nodes * dim_dof, tot_el_nodes * dim_dof))
            # cont_test += 1
            elK: np = elem.stiffness(ns, mat, elem.id)
            # .................................................................
            for node_row in range(tot_el_nodes):
                pos_row: int = pos[node_row]
                for node_col in range(tot_el_nodes):
                    pos_col: int = pos[node_col]
                    for i in range(dim_dof):
                        row: int = dim_dof * pos_row + i
                        row_el: int = dim_dof * node_row + i
                        for j in range(dim_dof):
                            col: int = dim_dof * pos_col + j
                            col_el: int = dim_dof * node_col + j
                            K[row, col] += elK[row_el, col_el]
        return K

    # ---------------------------------------------------------------------------
    def assembly_loads(self, nodes: list[Node]) -> tuple[np, np]:
        dim: int = len(nodes)
        if dim <= 0:   #se non vi è la lista dei nodi esce
            return (None, None)
        NDOF = nodes[0].NDOF
        dim = dim * NDOF
        f: np = np.zeros(dim)
        fix: np = np.zeros(dim)
        cont: int = 0
        for n, node in enumerate(nodes): #n=posizione nodo, node=elemento associato alla posizione
            for i, dof in enumerate(node.dof):
                if node.fix[i] == 1:
                    f[cont] = node.dof[i]
                    fix[cont] = node.fix[i]
                cont += 1

        return (f, fix)

    # ---------------------------------------------------------------------------
    def linear_solver(self, K, f, fix) -> np:
        print(f'Matrice K\n'
              f'{K}\n')
        K_inv: np = np.zeros((len(K), len(K)))[0]
        delete_index = np.where(fix == 1)
        print("fix:", fix)
        print("old K shape:", K.shape)
        K_rid = np.delete(K, delete_index, axis=0)  # Colonne
        K_rid = np.delete(K_rid, delete_index, axis=1)  # Righe
        f_rid = np.delete(f, delete_index, axis=0)  # Colonne f
        print(f'Matrice K ridotta\n'
              f'{K_rid}\n')
        print("K_rid shape:", K_rid.shape)
        if np.linalg.det(K_rid) == 0:
            print("WARNING: K_rid matrix is singular")
            exit()
        K_inv = np.linalg.inv(K_rid)  # inversione matrice K
        print(f'Matrice Inversa K_inv:\n'
              f'{K_inv}\n')
        return(K_inv, f_rid)

    # ---------------------------------------------------------------------------