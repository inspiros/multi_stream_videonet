from .c3d import *
from .densenet import *
from .mc3d import *
from .mobilenet import *
from .mobilenetv2 import *
from .r2plus1d import *
from .r3d import *
from .resnext import *
from .shufflenet import *
from .shufflenetv2 import *
from .wide_resnet import *
from .wide_resnext import *

__all__ = ['get_arch', 'get_model']


def get_arch(arch_name):
    try:
        return eval(arch_name)
    except:
        raise ValueError(f'architecture {arch_name} not found.')


def get_model(arch_name, *args, **kwargs):
    return get_arch(arch_name)(*args, **kwargs)
