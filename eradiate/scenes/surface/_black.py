import attr

from ._core import Surface, SurfaceFactory
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@SurfaceFactory.register("black")
@parse_docs
@attr.s
class BlackSurface(Surface):
    """
    Black surface scene element [:factorykey:`black`].

    This class creates a square surface with a black BRDF attached.
    """

    def bsdfs(self, ctx: KernelDictContext = None):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.0},
            }
        }
