"""WRF-Chem-inspired reduced-order coupled atmospheric model for Delhi NCR.

WRF-CHEM DELHI V2 - Smart India Hackathon prototype.

System honesty: this package is a physics-informed *reduced-order* coupled
atmospheric chemistry model. It is NOT operational WRF-Chem. This
architecture provides a swappable ingress point (``wrf_chem_delhi.adapter``)
so an actual WRF-Chem installation can be connected later with zero changes
to the rest of the stack.
"""

__version__ = "2.0.0"
MODEL_NAME = "WRF-Chem-inspired reduced-order coupled atmospheric model"
V2 = True