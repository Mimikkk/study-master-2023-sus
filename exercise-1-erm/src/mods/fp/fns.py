from typing import Iterable

def exhaust(iterable: Iterable[map]) -> tuple:
  if isinstance(iterable, map): return tuple(exhaust(element) for element in iter(iterable))
  return iterable
