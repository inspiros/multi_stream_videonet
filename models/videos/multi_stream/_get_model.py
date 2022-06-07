from .multi_stream_movinet import *

__all__ = ['get_arch', 'get_model']


def get_arch(arch_name):
    try:
        return eval(arch_name)
    except:
        raise ValueError(f'architecture {arch_name} not found.')


def get_model(arch_name, *args, **kwargs):
    return get_arch(arch_name)(*args, **kwargs)
