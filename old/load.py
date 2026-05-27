class Load:

    def __init__(self, ID: int = 0, f: list[float]=0, id_tab: float = 0) -> None:
        self.id = ID
        self.f = f
        self.id_tab = id_tab
        if ID is not None:
            self.id = ID