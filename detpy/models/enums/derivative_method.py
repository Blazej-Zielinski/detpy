from enum import Enum

class DerivativeMethod(Enum):
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    AUTOMATIC = "automatic"