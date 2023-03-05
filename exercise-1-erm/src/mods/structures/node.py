from typing import Iterable, overload
from dataclasses import dataclass

@dataclass
class Node(object):
  feature: int = None
  threshold: float = None
  left: 'Node' = None
  right: 'Node' = None
  variance: float = None
  value: float = None
