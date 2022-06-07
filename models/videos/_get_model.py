from .c3d import *
from .densenet3d import *
from .efficientnet3d import *
from .mc3d import *
from .mobilenet3d import *
from .mobilenet3d_v2 import *
from .movinet import *
from .r2plus1d import *
from .r3d import *
from .resnext3d import *
from .shufflenet3d import *
from .shufflenet3d_v2 import *
from .wide_resnet3d import *
from .wide_resnext3d import *

__all__ = ['get_arch', 'get_model']


def get_arch(arch_name):
    try:
        return eval(arch_name)
    except:
        raise ValueError(f'architecture {arch_name} not found.')


def get_model(arch_name, *args, **kwargs):
    return get_arch(arch_name)(*args, **kwargs)
