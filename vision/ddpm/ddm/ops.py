import mindspore
from mindspore import ops, Tensor, context
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import constexpr

gpu_target = (context.get_context("device_target") == "GPU")

def rsqrt(x):
    rsqrt_op = _get_cache_prim(ops.Rsqrt)()
    return rsqrt_op(x)

def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x*y))

def randint(low, high, size, dtype=mindspore.int32):
    uniform_int = _get_cache_prim(ops.UniformInt)()
    return uniform_int(size, Tensor(low, mindspore.int32), Tensor(high, mindspore.int32)).astype(dtype)

def random():
    uniform = _get_cache_prim(ops.UniformReal)()
    return uniform((1,))

def randn_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(x.shape).astype(dtype)

def randn(shape, dtype=None):
    if dtype is None:
        dtype = mindspore.float32
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(shape).astype(dtype)

def cumprod(input, dim, dtype=None):
    cumprod_op = _get_cache_prim(ops.CumProd)()
    output = cumprod_op(input, dim)
    if dtype:
        output = _get_cache_prim(ops.Cast)()(output, dtype)
    return output

def softmax(x, axis=-1):
    if gpu_target:
        softmax_ = _get_cache_prim(ops.Softmax)(axis=axis)
        return softmax_(x)
    exp_ = _get_cache_prim(ops.Exp)()
    reduce_sum_ = _get_cache_prim(ops.ReduceSum)(True)

    x_max = x.max(axis=axis, keepdims=True)
    x_exp = exp_(x - x_max)
    partion = reduce_sum_(x_exp, axis)
    return x_exp / partion

inf = float('inf')

@constexpr
def raise_value_error(info):
    raise ValueError(info)

@constexpr
def raise_runtime_error(info):
    raise RuntimeError(info)

@constexpr
def raise_type_error(info):
    raise TypeError(info)

def _check_dtype(d1, d2):
    if mindspore.float32 in (d1, d2):
        return mindspore.float32
    if d1 == d2:
        return d1
    raise ValueError('dtype is not supported.')

def dot(a, b):
    res_dtype = _check_dtype(a.dtype, b.dtype)
    ndim_a, ndim_b = a.ndim, b.ndim
    if ndim_a == 0 or ndim_b == 0:
        return ops.tensor_mul(a, b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = ops.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = ops.transpose(b, perm)

    if a.shape[-1] != b.shape[-1]:
        raise_value_error('shapes are not aligned')
    a_aligned = a.reshape(-1, a.shape[-1]).astype(mindspore.float32)
    b_aligned = b.reshape(-1, b.shape[-1]).astype(mindspore.float32)

    res = ops.matmul(a_aligned, b_aligned.T)
    res = res.reshape(a.shape[:-1] + b.shape[:-1])

    return res.astype(res_dtype)

def sqrt(x):
    return ops.sqrt(x.astype(mindspore.float32))

def reciprocal(x):
    if isinstance(x, Tensor):
        _reciprocal = _get_cache_prim(ops.Reciprocal)()
        return _reciprocal(x)
    return 1/x

@constexpr
def _check_axis(axis, ord, ndim):
    if axis is None:
        axis = tuple(range(ndim))
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)):
            return axis, True
        else:
            return axis, False
    else:
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, tuple):
            if len(axis) > 2:
                raise ValueError("Improper number of dimensions to norm.")
        else:
            raise ValueError(f'axis should be int or tuple but got {type(axis)}')
        return axis, False

@constexpr
def _check_ord(ord, axis):
    if len(axis) == 1:
        if isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
    elif len(axis) == 2:
        if ord not in [2, -2, 1, -1, inf, -inf, 'fro', 'f', 'nuc', None]:
            raise ValueError("Invalid norm order for matrices.")

def norm(x, ord=None, axis=None, keepdims=False):
    ndim = x.ndim
    # Normalize the `axis` argument to a tuple.
    axis, immediate = _check_axis(axis, ord, ndim)
    _check_ord(ord, axis)
    # Immediately handle some default, simple, fast, and common cases.
    if immediate:
        x = x.ravel()
        sqnorm = dot(x, x)
        ret = sqrt(sqnorm)
        if keepdims:
            ret = ret.reshape(ndim*[1])
        return ret

    if isinstance(ord, float):
        ord = int(ord)
        _lp_norm = _get_cache_prim(ops.LpNorm)(axis, ord, keepdims)
        return _lp_norm(x)

    if len(axis) == 1:
        if ord == inf:
            return ops.abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -inf:
            return ops.abs(x).min(axis=axis, keepdims=keepdims)
        elif ord is None:
            # special case for speedup
            conj = _get_cache_prim(ops.Conj)()
            s = conj(x) * x
            reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
            return sqrt(reduce_sum(s, axis=axis))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        else:
            absx = ops.abs(x)
            absx **= ord
            reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
            ret = reduce_sum(absx, axis=axis)
            ret **= reciprocal(ord)
            if ops.isnan(ret):
                return ops.zeros_like(ret)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, ndim)
        col_axis = normalize_axis_index(col_axis, ndim)
        if row_axis == col_axis:
            raise_value_error('Duplicate axes given.')

        if ord == inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in ['fro', 'f']:
            conj = _get_cache_prim(ops.Conj)()
            ret = sqrt(ops.reduce_sum((conj(x) * x), axis=axis))
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            conj = _get_cache_prim(ops.Conj)()
            ret = sqrt(ops.reduce_sum((conj(x) * x), axis=axis))
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        return None

def _multi_svd_norm(x, row_axis, col_axis, op):
    y = moveaxis(x.astype(mindspore.float32), (row_axis, col_axis), (-2, -1))
    if op == 'amax':
        result = ops.svd(y, compute_uv=False).max(axis=-1)
    elif op == 'amin':
        result = ops.svd(y, compute_uv=False).min(axis=-1)
    else:
        result = None
    return result

def normalize_axis_index(axis, ndim):
    if axis >= 0 and axis < ndim:
        return axis
    elif axis < 0 and axis >= -ndim:
        return ndim + axis
    else:
        return axis

def moveaxis(x, source, destination):
    perm = [i for i in range(x.ndim)]
    for s, d in zip(source, destination):
        tmp = perm[s]
        perm[s] = perm[d]
        perm[d] = tmp
    perm = tuple(perm)
    return ops.transpose(x, perm)

def clip_grad_norm(grads, max_norm: float, norm_type: float = 2.0):
    if isinstance(grads, mindspore.Tensor):
        grads = [grads]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return [], mindspore.Tensor(0., mindspore.float32)

    if norm_type == inf:
        norms = [grad.abs().max() for grad in grads]
        total_norm = norms[0] if len(norms) == 1 else ops.max(ops.stack(norms))
    else:
        norms = ()
        for grad in grads:
            norms += (norm(grad, norm_type),)
        total_norm = norm(ops.stack(norms), norm_type)
    # print(total_norm)
    clip_coef = ops.div(max_norm, (total_norm + ops.scalar_to_tensor(1e-6, mindspore.float32)))
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = clip_coef.clip(None, 1.0)
    new_grads = ()
    for grad in grads:
        new_grads += (ops.mul(grad, clip_coef_clamped),)
    return new_grads, total_norm
