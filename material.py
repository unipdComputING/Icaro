class Material:

    def __init__(self, ID: int = 0, dens: float = 0, cond: float = 0, cspec: float = 0) -> None:
        self.id = ID
        self.dens = dens
        self.cond = cond
        self.cspec = cspec
        if ID is not None:
            self.id = ID
