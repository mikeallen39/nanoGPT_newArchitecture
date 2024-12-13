@dataclass
class MambaConfig:
    d_model: int = 16
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2