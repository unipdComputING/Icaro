class Boundary_condition:
    def __init__(self, ID: int = 0, fix: int = 0, t: float = 0) -> None:
        self.id = id
        self.fix = fix
        self.t = t
        if ID is not None:
            self.id = ID