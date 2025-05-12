from enum import Enum

class BaseVectorSchema(Enum):
    RAND = 'rand'
    CURRENT = 'current'
    BEST = 'best'
