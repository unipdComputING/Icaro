import gmsh
import sys

gmsh.initialize()
gmsh.model.add("t1")

lc = 1

# PUNTI
gmsh.model.geo.addPoint(0,   0,   0, lc, 1)  # bottom-left
gmsh.model.geo.addPoint(10, 0,   0, lc, 2)  # bottom-right
gmsh.model.geo.addPoint(10, 30, 0, lc, 3)  # top-right
gmsh.model.geo.addPoint(0,   30, 0, lc, 4)  # top-left

# CURVE
gmsh.model.geo.addLine(1, 2, 1)  # bottom
gmsh.model.geo.addLine(2, 3, 2)  # right
gmsh.model.geo.addLine(3, 4, 3)  # top
gmsh.model.geo.addLine(4, 1, 4)  # left

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()


gmsh.model.addPhysicalGroup(2, [1], 1)
gmsh.model.setPhysicalName(2, 1, "mat_1")

#SET
gmsh.model.addPhysicalGroup(1, [1], 2)
gmsh.model.setPhysicalName(1, 2, "bottom")

gmsh.model.addPhysicalGroup(1, [2], 3)
gmsh.model.setPhysicalName(1, 3, "right")

gmsh.model.addPhysicalGroup(1, [3], 4)
gmsh.model.setPhysicalName(1, 4, "top")

gmsh.model.addPhysicalGroup(1, [4], 5)
gmsh.model.setPhysicalName(1, 5, "left")

# MESH
gmsh.model.mesh.generate(2)
gmsh.write("t1.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()