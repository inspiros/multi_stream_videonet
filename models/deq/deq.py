import platform
import warnings
from typing import Union, Mapping, Sequence, Callable, Any

import torch
import torch.autograd as autograd
import torch.nn as nn

from ._tensor_utils import *
from .solvers import *

__all__ = ['DEQBlock']

_IS_WINDOWS = platform.system() == 'Windows'
DEFAULT_DEQ_MODE = 'backward_hook' if not _IS_WINDOWS else 'safe_backward_hook'
DEFAULT_DEQ_AUTO_REMOVE_HOOK = not _IS_WINDOWS


# noinspection PyMethodOverriding
class DEQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                f,
                x0,
                solver,
                solver_kwargs):
        ctx.f = f
        ctx.solver = solver
        ctx.solver_kwargs = solver_kwargs

        result = solver(f,
                        x0,
                        **solver_kwargs)
        z_star = result[0]

        ctx.save_for_backward(z_star)
        return f(z_star)

    @staticmethod
    def backward(ctx, grad_output):
        f = ctx.f
        solver = ctx.solver
        solver_kwargs = ctx.solver_kwargs

        z_star, = ctx.saved_tensors
        with torch.enable_grad():
            new_z_star = f(z_star.requires_grad_(True))
        result = solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad_output,
                        torch.zeros_like(grad_output),
                        **solver_kwargs)
        new_z_star.backward(result[0], retain_graph=True)
        return None, None, None, None


# noinspection PyMethodOverriding
class SafeDEQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                f,
                x0,
                solver,
                solver_kwargs):
        ctx.f = f
        ctx.solver = solver
        ctx.solver_kwargs = solver_kwargs

        result = solver(f,
                        x0,
                        **solver_kwargs)
        z_star = result[0]

        ctx.save_for_backward(z_star)
        return f(z_star)

    @staticmethod
    def backward(ctx, grad_output):
        f = ctx.f
        solver = ctx.solver
        solver_kwargs = ctx.solver_kwargs

        z_star, = ctx.saved_tensors
        with torch.enable_grad():
            z_star = f(z_star)
            z0 = z_star.detach().requires_grad_(True)
            new_z_star = f(z0)
        result = solver(lambda y: autograd.grad(new_z_star, z0, y, retain_graph=True)[0] + grad_output,
                        torch.zeros_like(grad_output),
                        **solver_kwargs)
        z_star.backward(result[0], retain_graph=True)
        return None, None, None, None


def deq_function(f,
                 x0,
                 solver,
                 solver_kwargs):
    # fake grad required input
    x0.requires_grad_(True)
    return DEQFunction.apply(f, x0, solver, solver_kwargs)


def safe_deq_function(f,
                      x0,
                      solver,
                      solver_kwargs):
    # fake grad required input
    x0.requires_grad_(True)
    return SafeDEQFunction.apply(f, x0, solver, solver_kwargs)


class DEQBlock(nn.Module):

    def __init__(self,
                 f: Callable,
                 solver: Union[Callable, str] = anderson,
                 deq_mode: Union[str, bool, None] = DEFAULT_DEQ_MODE,
                 num_layers: int = 5,
                 auto_remove_hook: bool = DEFAULT_DEQ_AUTO_REMOVE_HOOK,
                 **solver_kwargs):
        super(DEQBlock, self).__init__()
        if isinstance(solver, str):
            try:
                solver = eval(solver)
            except NameError:
                raise ValueError(f'Undefined solver {solver}.')

        self.f = f
        self.solver = solver
        self.deq_mode = None
        self.num_layers = num_layers

        self._forward_impl = None
        self.set_deq_mode(deq_mode)

        self.solver_kwargs = solver_kwargs
        self.hook = None
        self.auto_remove_hook = auto_remove_hook

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                m.weight.data.normal_(0, 0.01)

    def set_deq_mode(self, deq_mode: Union[str, bool, None] = DEFAULT_DEQ_MODE):
        if isinstance(deq_mode, bool) and deq_mode:
            deq_mode = DEFAULT_DEQ_MODE

        if deq_mode == 'backward_hook':
            self._forward_impl = self._backward_hook_impl
        elif deq_mode == 'safe_backward_hook':
            self._forward_impl = self._safe_backward_hook_impl
        elif deq_mode == 'autograd':
            warnings.warn('Autograd Function DEQ implementation is experimental and causes extra '
                          'overheads in backward pass, but might be more efficient for inference.')
            self._forward_impl = self._autograd_impl
        elif deq_mode == 'safe_autograd':
            warnings.warn('Autograd Function DEQ implementation is experimental and causes extra '
                          'overheads in backward pass, but might be more efficient for inference.')
            self._forward_impl = self._safe_autograd_impl
        elif deq_mode == 'original':
            self._forward_impl = self._original_impl
        elif deq_mode in ['deterministic', False, None]:
            self._forward_impl = self._deterministic_impl
        else:
            raise ValueError(f'deq_mode {self.deq_mode} not recognized.')
        self.deq_mode = deq_mode

    def forward(self,
                x: Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]
                ) -> Union[torch.Tensor, Mapping[Any, torch.Tensor], Sequence[torch.Tensor]]:
        return self._forward_impl(x)

    def _deterministic_impl(self, x):
        z = x
        for layer_ind in range(self.num_layers):
            z = self.f(z, x)
        return z

    def _backward_hook_impl(self, x):
        pack_fn = get_default_collate_fn()
        unpack_fn = get_default_decollate_fn(x)
        f = lambda z: pack_fn(self.f(unpack_fn(z), x))

        with torch.no_grad():
            result = self.solver(f,
                                 torch.zeros_like(pack_fn(x)),
                                 **self.solver_kwargs)
            z_star = result[0]

        with torch.enable_grad():
            new_z_star = f(z_star.requires_grad_(True))

        def backward_hook(grad):
            if self.hook is not None and self.auto_remove_hook:
                self.hook.remove()
            # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
            result = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad,
                                 torch.zeros_like(grad),
                                 **self.solver_kwargs)
            return result[0]

        self.hook = new_z_star.register_hook(backward_hook)
        return unpack_fn(new_z_star)

    def _safe_backward_hook_impl(self, x):
        pack_fn = get_default_collate_fn()
        unpack_fn = get_default_decollate_fn(x)
        func = lambda z: pack_fn(self.f(unpack_fn(z), x))

        with torch.no_grad():
            result = self.solver(func,
                                 torch.zeros_like(pack_fn(x)),
                                 **self.solver_kwargs)
            z_star = result[0]

        with torch.enable_grad():
            z_star = func(z_star)
            # set up Jacobian vector product (without additional forward calls)
            z0 = z_star.detach().requires_grad_(True)
            new_z_star = func(z0)

        def backward_hook(grad):
            if self.hook is not None and self.auto_remove_hook:
                self.hook.remove()
            # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
            result = self.solver(lambda y: autograd.grad(new_z_star, z0, y, retain_graph=True)[0] + grad,
                                 torch.zeros_like(grad),
                                 **self.solver_kwargs)
            return result[0]

        self.hook = z_star.register_hook(backward_hook)
        return unpack_fn(z_star)

    def _original_impl(self, x):
        if not torch.is_tensor(x):
            raise ValueError('Original DEQ implementation method only allows Tensor input.')

        with torch.no_grad():
            result = self.solver(lambda z: self.f(z, x),
                                 torch.zeros_like(x),
                                 **self.solver_kwargs)
            z_star = result[0]

        with torch.enable_grad():
            z_star = self.f(z_star, x)
            # set up Jacobian vector product (without additional forward calls)
            z0 = z_star.clone().detach().requires_grad_(True)
            f0 = self.f(z0, x)

        def backward_hook(grad):
            if self.hook is not None and self.auto_remove_hook:
                self.hook.remove()
            # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
            result = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                 torch.zeros_like(grad),
                                 **self.solver_kwargs)
            return result[0]

        z_star.register_hook(backward_hook)
        return z_star

    def _autograd_impl(self, x):
        pack_fn = get_default_collate_fn()
        unpack_fn = get_default_decollate_fn(x)
        f = lambda z: pack_fn(self.f(unpack_fn(z), x))
        new_z_star = deq_function(f,
                                  torch.zeros_like(pack_fn(x)),
                                  self.solver,
                                  self.solver_kwargs)
        return unpack_fn(new_z_star)

    def _safe_autograd_impl(self, x):
        pack_fn = get_default_collate_fn()
        unpack_fn = get_default_decollate_fn(x)
        f = lambda z: pack_fn(self.f(unpack_fn(z), x))
        new_z_star = safe_deq_function(f,
                                       torch.zeros_like(pack_fn(x)),
                                       self.solver,
                                       self.solver_kwargs)
        return unpack_fn(new_z_star)
