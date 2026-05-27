from table import Table
class Amp:

  def __init__(self, name: str = None, time: list[float] = None, intensity: list[float] = None) -> None:


    self.name: str = name
    self.time: list = time
    self.intensity: list = intensity
    self.data: Table = Table(name=name)


    if self.name is None:
      self.name = 'amp_test'
    if self.time is None:
      self.time = []
    if self.intensity is None:
      self.intensity = []

  def get_time(self) -> list[int]:
    return self.time
  def get_intensity(self) -> list[int]:
    return self.intensity
  def get_name(self) -> str:
    return self.name

  def get_amp(self) -> tuple[str, list[int], list[int]]:
    return self.name, self.time, self.intensity,