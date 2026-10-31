"""Our own NF4 dequantization kernel (Triton), used in place of the bitsandbytes CUDA kernel.

Why: every 4-bit layer (7 per decoder layer, 28 layers) unpacks its NF4 weight to fp16 once in the
forward pass and once in the backward pass: 389 launches per step, about 4% of the GPU time in our
profile (`kDequantizeBlockwise`).

How NF4 is stored (bitsandbytes, see csrc/kernels.cu and csrc/ops.cu in bitsandbytes 0.50.2):
- each byte holds two 4-bit codes: element 2*i is the high nibble, element 2*i+1 the low nibble;
- each code indexes a fixed table of 16 values (the "NormalFloat" code book);
- every `blocksize` consecutive elements share one fp32 scale (`absmax`).
So:  weight[e] = TABLE[code(e)] * absmax[e // blocksize],  rounded to the output dtype.
We do the same float32 multiply and the same rounding, so the result is bit-identical.

The idea and the data layout come from bitsandbytes (MIT); the kernel code here is our own.
"""

import torch
import triton
import triton.language as tl

# NF4 code book, the values of `nf4_dequantization_lut` in bitsandbytes csrc/kernels.cu.
NF4_TABLE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)

ELEMENTS_PER_PROGRAM = 2048  # each Triton program writes this many output values


@triton.jit
def nf4_unpack_scale_kernel(packed_ptr, absmax_ptr, table_ptr, out_ptr, n_elements,
                            QBLOCK: tl.constexpr, BLOCK: tl.constexpr):
    """One program: BLOCK consecutive output values, read from BLOCK // 2 packed bytes."""
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # output element index
    inside = idx < n_elements
    byte = tl.load(packed_ptr + (idx >> 1), mask=inside, other=0).to(tl.int32)
    # even element -> high 4 bits, odd element -> low 4 bits
    code = tl.where((idx & 1) == 0, byte >> 4, byte & 0xF)
    value = tl.load(table_ptr + code, mask=inside, other=0.0)            # fp32 code book entry
    scale = tl.load(absmax_ptr + idx // QBLOCK, mask=inside, other=0.0)  # fp32 block scale
    tl.store(out_ptr + idx, (value * scale).to(out_ptr.dtype.element_ty), mask=inside)


_tables = {}


def _nf4_table(device):
    table = _tables.get(device)
    if table is None:
        table = torch.tensor(NF4_TABLE, dtype=torch.float32, device=device)
        _tables[device] = table
    return table


def nf4_dequantize(packed, absmax, blocksize, out):
    """Fill `out` (any float dtype, contiguous) with the dequantized NF4 weight."""
    packed = packed.contiguous().view(torch.uint8)
    absmax = absmax.contiguous()
    n = out.numel()
    grid = (triton.cdiv(n, ELEMENTS_PER_PROGRAM),)
    nf4_unpack_scale_kernel[grid](packed, absmax, _nf4_table(out.device), out, n,
                                  QBLOCK=blocksize, BLOCK=ELEMENTS_PER_PROGRAM, num_warps=4)
    return out


def patch_bnb_dequant():
    """Route bitsandbytes' 4-bit dequantization through our kernel.

    In bitsandbytes 0.50.2 every CUDA path that unpacks a 4-bit weight (the forward pass of
    MatMul4Bit through `_dequant_linear_fallback`, and `dequantize_4bit` used in its backward pass)
    calls the module function `_dequantize_4bit_impl` by name, so replacing that one function is
    enough. Anything we do not handle (FP4, odd dtypes) goes to the original.
    """
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    original = bnb_cuda_ops._dequantize_4bit_impl
    if getattr(original, "is_soup_triton", False):
        return

    def dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
        if (quant_type != "nf4" or not A.is_cuda or absmax.dtype != torch.float32
                or dtype not in (torch.float16, torch.float32, torch.bfloat16)
                or out.dtype != dtype or not out.is_contiguous()):
            return original(A, absmax, blocksize, quant_type, dtype, out)
        nf4_dequantize(A, absmax, blocksize, out)
        return None

    dequantize_4bit_impl.is_soup_triton = True
    dequantize_4bit_impl.original = original
    bnb_cuda_ops._dequantize_4bit_impl = dequantize_4bit_impl


def unpatch_bnb_dequant():
    """Undo `patch_bnb_dequant` (used by the tests)."""
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    current = bnb_cuda_ops._dequantize_4bit_impl
    if getattr(current, "is_soup_triton", False):
        bnb_cuda_ops._dequantize_4bit_impl = current.original
