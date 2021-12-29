import collections.abc
import math
import operator
import sys
from functools import reduce
from typing import Union, Mapping, Dict, Sequence, Any, Callable

import torch

__all__ = [
    'prod',
    'identity',
    'clone',
    'zeros_like',
    'transform',
    'tensor_options',
    'get_default_collate_fn',
    'get_default_decollate_fn']

if sys.version_info.major <= 3 and sys.version_info.minor < 8:
    def prod(x):
        return reduce(operator.mul, x, 1)
else:
    prod = math.prod


def identity(x: Any) -> Any:
    return x


def clone(x: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]
          ) -> Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]:
    elem_type = type(x)
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    elif isinstance(x, collections.abc.Mapping):
        return elem_type({k: v.detach().clone() for k, v in x.items()})
    elif isinstance(x, collections.abc.Sequence):
        return elem_type([v.detach().clone() for v in x])
    raise TypeError(f"x must be tensor or collection of tensors "
                    f"(dicts or lists); found {elem_type}.")


def transform(
        x: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]],
        f: Callable = identity,
        inplace: bool = False,
) -> Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]:
    elem_type = type(x)
    if isinstance(x, torch.Tensor):
        return f(x)
    elif isinstance(x, collections.abc.MutableMapping) and not inplace:
        for k in x.keys():
            x[k] = f(x[k])
        return x
    elif isinstance(x, collections.abc.Mapping):
        return elem_type({k: f(v) for k, v in x.items()})
    elif isinstance(x, collections.abc.MutableSequence) and not inplace:
        for i in range(len(x)):
            x[i] = f(x[i])
        return x
    elif isinstance(x, collections.abc.Sequence):
        return elem_type([f(v) for v in x])
    raise TypeError(f"x must be tensor or collection of tensors "
                    f"(dicts or lists); found {elem_type}.")


def zeros_like(
        x: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]
) -> Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]:
    elem_type = type(x)
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif isinstance(x, collections.abc.Mapping):
        return elem_type({k: torch.zeros_like(v) for k, v in x.items()})
    elif isinstance(x, collections.abc.Sequence):
        return elem_type([torch.zeros_like(v) for v in x])
    raise TypeError(f"x must be tensor or collection of tensors "
                    f"(dicts or lists); found {elem_type}.")


def tensor_options(
        x: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]
) -> Dict:
    elem_type = type(x)
    if isinstance(x, torch.Tensor):
        return dict(dtype=x.dtype, device=x.device)
    elif isinstance(x, collections.abc.Mapping):
        x0 = next(iter(x.values()))
        return dict(dtype=x0.dtype, device=x0.device)
    elif isinstance(x, collections.abc.Sequence):
        x0 = next(iter(x))
        return dict(dtype=x0.dtype, device=x0.device)
    raise TypeError(f"x must be tensor or collection of tensors "
                    f"(dicts or lists); found {elem_type}.")


def get_default_collate_fn() -> Callable:
    def default_collate(
            x0: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        elem_type = type(x0)
        if isinstance(x0, torch.Tensor):
            return x0.flatten(1)
        elif isinstance(x0, collections.abc.Mapping):
            return torch.cat([v.flatten(1) for k, v in x0.items()], dim=1)
        elif isinstance(x0, collections.abc.Sequence):
            return torch.cat([v.flatten(1) for v in x0], dim=1)
        raise TypeError(f"default_collate: x0 must be tensor or collection of tensors "
                        f"(dicts or lists); found {elem_type}.")

    return default_collate


def get_default_decollate_fn(x0: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]) -> Callable:
    elem_type = type(x0)
    case = -1
    dims = flattened_dims = keys = None
    n_elems = 0
    if isinstance(x0, torch.Tensor):
        case = 0
        n_elems = 1
        dims = x0.size()
        flattened_dims = None
    elif isinstance(x0, collections.abc.Mapping):
        case = 1
        n_elems = len(x0)
        keys = list(x0.keys())
        dims = [v.size() for v in x0.values()]
        flattened_dims = [prod(v.size()[1:]) for v in x0.values()]
    elif isinstance(x0, collections.abc.Sequence):
        case = 2
        n_elems = len(x0)
        dims = [v.size() for v in x0]
        flattened_dims = [prod(v.size()[1:]) for v in x0]
    else:
        raise TypeError(f"default_decollate: x0 must be tensor or collection of tensors "
                        f"(dicts or lists); found {elem_type}.")
    del x0

    def default_decollate(
            z: torch.Tensor
    ) -> Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]:
        if case == 0:
            return z.view(dims)
        elif case == 1:
            return elem_type(
                {k: z[:, sum(flattened_dims[:i]):sum(flattened_dims[:i + 1])].view_as(dims[i])
                 for i, k in enumerate(keys)})
        elif case == 2:
            return elem_type([z[:, sum(flattened_dims[:i]):sum(flattened_dims[:i + 1])].view(dims[i])
                              for i in range(n_elems)])

    return default_decollate
