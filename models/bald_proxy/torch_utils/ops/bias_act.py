import os

import numpy as np
import torch
from ... import dnnlib
from .. import custom_ops
from .. import misc


activation_funcs = {
    "linear": dnnlib.EasyDict(func=lambda x, **_: x, def_alpha=0, def_gain=1, cuda_idx=1, ref="", has_2nd_grad=False),
    "relu": dnnlib.EasyDict(
        func=lambda x, **_: torch.nn.functional.relu(x),
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=2,
        ref="y",
        has_2nd_grad=False,
    ),
    "lrelu": dnnlib.EasyDict(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2,
        def_gain=np.sqrt(2),
        cuda_idx=3,
        ref="y",
        has_2nd_grad=False,
    ),
    "tanh": dnnlib.EasyDict(func=lambda x, **_: torch.tanh(x), def_alpha=0, def_gain=1, cuda_idx=4, ref="y", has_2nd_grad=True),
    "sigmoid": dnnlib.EasyDict(
        func=lambda x, **_: torch.sigmoid(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=5,
        ref="y",
        has_2nd_grad=True,
    ),
    "elu": dnnlib.EasyDict(
        func=lambda x, **_: torch.nn.functional.elu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=6,
        ref="y",
        has_2nd_grad=True,
    ),
    "selu": dnnlib.EasyDict(
        func=lambda x, **_: torch.nn.functional.selu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=7,
        ref="y",
        has_2nd_grad=True,
    ),
    "softplus": dnnlib.EasyDict(
        func=lambda x, **_: torch.nn.functional.softplus(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=8,
        ref="y",
        has_2nd_grad=True,
    ),
    "swish": dnnlib.EasyDict(
        func=lambda x, **_: torch.sigmoid(x) * x,
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=9,
        ref="x",
        has_2nd_grad=True,
    ),
}


_plugin = None


def _init():
    global _plugin
    if _plugin is None:
        sources = [
            os.path.join(os.path.dirname(__file__), "bias_act.cpp"),
            os.path.join(os.path.dirname(__file__), "bias_act.cu"),
        ]
        # Some packaged environments ship empty placeholder CUDA sources for bald_proxy.
        # Fall back to the PyTorch reference implementation instead of failing import/build.
        if any((not os.path.isfile(path)) or os.path.getsize(path) == 0 for path in sources):
            return False
        try:
            _plugin = custom_ops.get_plugin(
                module_name="bias_act_plugin",
                sources=sources,
                extra_cuda_cflags=["--use_fast_math"],
            )
        except Exception:
            _plugin = False
    return bool(_plugin)


def bias_act(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None, impl="cuda"):
    assert isinstance(x, torch.Tensor)
    assert impl in ["ref", "cuda"]
    if impl == "cuda" and x.device.type == "cuda" and _init():
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)


def _bias_act_ref(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None):
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    x = spec.func(x, alpha=alpha)
    if gain != 1:
        x = x * gain
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)
    return x


_bias_act_cuda_cache = {}


def _bias_act_cuda(dim=1, act="linear", alpha=None, gain=None, clamp=None):
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    key = (dim, act, alpha, gain, clamp)
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]

    class BiasActCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, b):
            if x.ndim > 8:
                return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)

            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride(1) == 1 else torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else x.new_empty([0])
            y = x
            if act != "linear" or gain != 1 or clamp >= 0 or b is not None:
                y = _plugin.bias_act(x, b, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(x, b, y)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, b, y = ctx.saved_tensors
            dx = None
            db = None

            if ctx.needs_input_grad[0]:
                dx = dy
                if act != "linear" or gain != 1 or clamp >= 0:
                    dx = _plugin.bias_act(dy.contiguous(memory_format=ctx.memory_format), b, y, dim, spec.cuda_idx, alpha, gain, clamp)

            if ctx.needs_input_grad[1] and b.numel() != 0:
                db = dy
                if act != "linear" or gain != 1 or clamp >= 0:
                    db = _plugin.bias_act(dy.contiguous(), b, y, dim, spec.cuda_idx, alpha, gain, clamp)
                if db.ndim > 1:
                    db = db.sum([i for i in range(db.ndim) if i != dim])

            return dx, db

    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda


@misc.profiled_function
def _plugin_bias_act(x, b, y, dim, act, alpha, gain, clamp):
    return _plugin.bias_act(x, b, y, dim, act, alpha, gain, clamp)
