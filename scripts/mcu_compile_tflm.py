#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""MCU model compilation script (TFLM backend).

This script only supports TFLite models compiled
via TensorFlow Lite Micro + Vela (Ethos-U55) / CMSIS-NN.

Licensing note (this compiler is Apache-2.0):
  - Generated model.h / model.cc / model_data.h wrapper code, and the
    non-data portions of model_data.c, are simple wrapper scripts and are
    released under BSD-3-Clause (see the SPDX tags emitted in each file).
  - The model_data.c byte array (the serialized/quantized/Vela-transformed
    model itself) retains the customer model's original license; this
    script does not relicense customer model weights.
  - TFLM, CMSIS-NN, and Ethos-U runtime components are supplied separately
    via FSP under their own respective licenses.
"""

__version__ = "1.0.0"

import os
import sys
import json
import struct
import shutil
import subprocess
import random
import gc
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from argparse import ArgumentParser, Namespace
import numpy as np

# Force UTF-8 on stdout/stderr so status messages containing symbols (checkmarks,
# warning signs) print correctly regardless of the host console's default
# encoding (e.g. GBK, Shift-JIS) or whether output is piped/redirected.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Default OSPI threshold (MB) - models larger than this auto-enable OSPI for NPU
DEFAULT_OSPI_THRESHOLD_MB = 0.8
DEFAULT_VELA_OPTIMIZATION = 'Performance'

# Embedded Vela configuration for RA8P1 (avoids shipping a separate .ini file)
VELA_RA8P1_CONFIG = """\
; Vela configuration for Renesas RA8P1 (Ethos-U55-256)
[System_Config.RA8P1]
core_clock=500e6
axi0_port=Sram
axi1_port=OnChipFlash
Sram_clock_scale=0.465
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
Sram_max_reads=4
Sram_max_writes=4
OnChipFlash_clock_scale=0.465
OnChipFlash_burst_length=128
OnChipFlash_read_latency=64
OnChipFlash_write_latency=0
OnChipFlash_max_reads=2
OnChipFlash_max_writes=0

[Memory_Mode.Sram_Only]
const_mem_area=Axi0
arena_mem_area=Axi0
cache_mem_area=Axi0
arena_cache_size=4194304

[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0
arena_cache_size=4194304
"""


# =============================================================================
# Data Classes
# =============================================================================

class Status(Enum):
    """Model processing status."""
    IN_PROGRESS = 1
    ERROR_DEPLOY = 5
    ERROR_VALIDATION = 7
    SUCCESS = 8


@dataclass
class CompileResult:
    """Result of compiling a single model."""
    model_name: str
    model_path: Path
    output_dir: Path
    status: Status = Status.IN_PROGRESS
    last_error: str = ''
    needs_external_memory: bool = False

# =============================================================================
# Warning message
# =============================================================================

def print_warning(message):
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}{message}{RESET}")

# =============================================================================
# Model Detection
# =============================================================================

def get_model_format(model_path: Path) -> str:
    """Validate the TFLite input format from the file extension."""
    suffix = model_path.suffix.lower()
    if suffix == '.tflite':
        return 'tflite'
    else:
        raise ValueError(
            f"Unsupported model format: {suffix}. TFLM accepts only .tflite input; "
            "convert other model formats to TFLite first."
        )


def _all_io_quantized(interpreter) -> bool:
    """True when every model input and output is an INT8/UINT8 tensor."""
    io_details = interpreter.get_input_details() + interpreter.get_output_details()
    return bool(io_details) and all(
        d['dtype'] in [np.int8, np.uint8] for d in io_details
    )


def is_model_quantized(model_path: Path) -> bool:
    """
    Check if a model is already quantized (INT8).
    
    For TFLite: Use TensorFlow Lite interpreter to check tensor dtypes
    """
    model_format = get_model_format(model_path)

    if model_format == 'tflite':
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()

            tensor_details = interpreter.get_tensor_details()
            total_count = len(tensor_details)
            int8_count = sum(
                1 for t in tensor_details if t['dtype'] in [np.int8, np.uint8]
            )

            # The body of the model is the reliable signal. A genuinely
            # quantized network stores its weights and activations as
            # INT8/UINT8, so they dominate the tensor list even when INT32
            # biases are counted. Hybrid models with float I/O but quantized
            # internals are also caught here.
            #
            # I/O dtype alone is not sufficient: an otherwise-FP32 model can
            # end in a single INT8 output tensor, and treating that as proof of
            # quantization made --quantize a silent no-op and handed raw FP32
            # to Vela, which then left every operator on the CPU.
            if total_count and int8_count > total_count * 0.5:
                if not _all_io_quantized(interpreter):
                    print(f"  Detected hybrid quantized model: "
                          f"{int8_count}/{total_count} INT8 tensors")
                return True

            return False
        except Exception as e:
            print(f"  Warning: Could not check TFLite quantization: {e}")
            # Fallback to filename check
            if 'int8' in model_path.name.lower() or 'quant' in model_path.name.lower():
                print(f"  Fallback: Detected INT8 via filename: {model_path.name}")
                return True
            return False
    
    
    return False


# =============================================================================
# Memory Calculation
# =============================================================================

def needs_external_memory(memory_bytes: int,
                          threshold_mb: float = DEFAULT_OSPI_THRESHOLD_MB) -> bool:
    """Check if model needs external memory based on size threshold."""
    threshold_bytes = threshold_mb * 1024 * 1024
    return memory_bytes > threshold_bytes


# =============================================================================
# TFLM Backend - TFLite Flatbuffer Parsing
# =============================================================================

# TFLite BuiltinOperator enum → TFLM C++ op resolver name.
# None means the FSP TFLM release cannot run the operator (no resolver
# method, or no kernel source shipped). An FSP/TFLM upgrade can invalidate
# this table; verify it against the target project's TFLM release.
TFLITE_BUILTIN_OP_MAP = {
    0: ('ADD', 'AddAdd'),
    1: ('AVERAGE_POOL_2D', 'AddAveragePool2D'),
    2: ('CONCATENATION', 'AddConcatenation'),
    3: ('CONV_2D', 'AddConv2D'),
    4: ('DEPTHWISE_CONV_2D', 'AddDepthwiseConv2D'),
    5: ('DEPTH_TO_SPACE', 'AddDepthToSpace'),
    6: ('DEQUANTIZE', 'AddDequantize'),
    7: ('EMBEDDING_LOOKUP', 'AddEmbeddingLookup'),
    8: ('FLOOR', 'AddFloor'),
    9: ('FULLY_CONNECTED', 'AddFullyConnected'),
    10: ('HASHTABLE_LOOKUP', None),
    11: ('L2_NORMALIZATION', 'AddL2Normalization'),
    12: ('L2_POOL_2D', 'AddL2Pool2D'),
    13: ('LOCAL_RESPONSE_NORMALIZATION', None),
    14: ('LOGISTIC', 'AddLogistic'),
    15: ('LSH_PROJECTION', None),
    16: ('LSTM', None),
    17: ('MAX_POOL_2D', 'AddMaxPool2D'),
    18: ('MUL', 'AddMul'),
    19: ('RELU', 'AddRelu'),
    20: ('RELU_N1_TO_1', None),
    21: ('RELU6', 'AddRelu6'),
    22: ('RESHAPE', 'AddReshape'),
    23: ('RESIZE_BILINEAR', 'AddResizeBilinear'),
    24: ('RNN', None),
    25: ('SOFTMAX', 'AddSoftmax'),
    26: ('SPACE_TO_DEPTH', 'AddSpaceToDepth'),
    27: ('SVDF', 'AddSvdf'),
    28: ('TANH', 'AddTanh'),
    29: ('CONCATENATION_EMBED', None),
    30: ('SKIP_GRAM', None),
    31: ('CALL', None),
    32: ('CUSTOM', None),
    33: ('EMBEDDING_LOOKUP_SPARSE', None),
    34: ('PAD', 'AddPad'),
    35: ('UNIDIRECTIONAL_SEQUENCE_RNN', None),
    36: ('GATHER', 'AddGather'),
    37: ('BATCH_TO_SPACE_ND', 'AddBatchToSpaceNd'),
    38: ('SPACE_TO_BATCH_ND', 'AddSpaceToBatchNd'),
    39: ('TRANSPOSE', 'AddTranspose'),
    40: ('MEAN', 'AddMean'),
    41: ('SUB', 'AddSub'),
    42: ('DIV', 'AddDiv'),
    43: ('SQUEEZE', 'AddSqueeze'),
    44: ('UNIDIRECTIONAL_SEQUENCE_LSTM', 'AddUnidirectionalSequenceLSTM'),
    45: ('STRIDED_SLICE', 'AddStridedSlice'),
    46: ('BIDIRECTIONAL_SEQUENCE_RNN', None),
    47: ('EXP', 'AddExp'),
    48: ('TOPK_V2', None),
    49: ('SPLIT', 'AddSplit'),
    50: ('LOG_SOFTMAX', 'AddLogSoftmax'),
    51: ('DELEGATE', None),
    52: ('BIDIRECTIONAL_SEQUENCE_LSTM', None),
    53: ('CAST', 'AddCast'),
    54: ('PRELU', 'AddPrelu'),
    55: ('MAXIMUM', 'AddMaximum'),
    56: ('ARG_MAX', 'AddArgMax'),
    57: ('MINIMUM', 'AddMinimum'),
    58: ('LESS', 'AddLess'),
    59: ('NEG', 'AddNeg'),
    60: ('PADV2', 'AddPadV2'),
    61: ('GREATER', 'AddGreater'),
    62: ('GREATER_EQUAL', 'AddGreaterEqual'),
    63: ('LESS_EQUAL', 'AddLessEqual'),
    64: ('SELECT', None),
    65: ('SLICE', 'AddSlice'),
    66: ('SIN', 'AddSin'),
    67: ('TRANSPOSE_CONV', 'AddTransposeConv'),
    68: ('SPARSE_TO_DENSE', None),
    69: ('TILE', None),
    70: ('EXPAND_DIMS', 'AddExpandDims'),
    71: ('EQUAL', 'AddEqual'),
    72: ('NOT_EQUAL', 'AddNotEqual'),
    73: ('LOG', 'AddLog'),
    74: ('SUM', 'AddSum'),
    75: ('SQRT', 'AddSqrt'),
    76: ('RSQRT', 'AddRsqrt'),
    77: ('SHAPE', 'AddShape'),
    78: ('POW', None),
    79: ('ARG_MIN', 'AddArgMin'),
    80: ('FAKE_QUANT', None),
    81: ('REDUCE_PROD', None),
    82: ('REDUCE_MAX', 'AddReduceMax'),
    83: ('PACK', 'AddPack'),
    84: ('LOGICAL_OR', 'AddLogicalOr'),
    85: ('ONE_HOT', None),
    86: ('LOGICAL_AND', 'AddLogicalAnd'),
    87: ('LOGICAL_NOT', 'AddLogicalNot'),
    88: ('UNPACK', 'AddUnpack'),
    89: ('REDUCE_MIN', 'AddReduceMin'),
    90: ('FLOOR_DIV', 'AddFloorDiv'),
    91: ('REDUCE_ANY', None),
    92: ('SQUARE', 'AddSquare'),
    93: ('ZEROS_LIKE', 'AddZerosLike'),
    94: ('FILL', 'AddFill'),
    95: ('FLOOR_MOD', 'AddFloorMod'),
    96: ('RANGE', None),
    97: ('RESIZE_NEAREST_NEIGHBOR', 'AddResizeNearestNeighbor'),
    98: ('LEAKY_RELU', 'AddLeakyRelu'),
    99: ('SQUARED_DIFFERENCE', 'AddSquaredDifference'),
    100: ('MIRROR_PAD', 'AddMirrorPad'),
    101: ('ABS', 'AddAbs'),
    102: ('SPLIT_V', 'AddSplitV'),
    103: ('UNIQUE', None),
    104: ('CEIL', 'AddCeil'),
    105: ('REVERSE_V2', None),
    106: ('ADD_N', 'AddAddN'),
    107: ('GATHER_ND', 'AddGatherNd'),
    108: ('COS', 'AddCos'),
    109: ('WHERE', None),
    110: ('RANK', None),
    111: ('ELU', 'AddElu'),
    112: ('REVERSE_SEQUENCE', None),
    113: ('MATRIX_DIAG', None),
    114: ('QUANTIZE', 'AddQuantize'),
    115: ('MATRIX_SET_DIAG', None),
    116: ('ROUND', 'AddRound'),
    117: ('HARD_SWISH', 'AddHardSwish'),
    118: ('IF', 'AddIf'),
    119: ('WHILE', 'AddWhile'),
    120: ('NON_MAX_SUPPRESSION_V4', None),
    121: ('NON_MAX_SUPPRESSION_V5', None),
    122: ('SCATTER_ND', None),
    123: ('SELECT_V2', 'AddSelectV2'),
    124: ('DENSIFY', None),
    125: ('SEGMENT_SUM', None),
    126: ('BATCH_MATMUL', 'AddBatchMatMul'),
    127: ('PLACEHOLDER_FOR_GREATER_OP_CODES', None),
    128: ('CUMSUM', 'AddCumSum'),
    129: ('CALL_ONCE', 'AddCallOnce'),
    130: ('BROADCAST_TO', 'AddBroadcastTo'),
    131: ('RFFT2D', None),
    132: ('CONV_3D', None),
    133: ('IMAG', None),
    134: ('REAL', None),
    135: ('COMPLEX_ABS', None),
    136: ('HASHTABLE', None),
    137: ('HASHTABLE_FIND', None),
    138: ('HASHTABLE_IMPORT', None),
    139: ('HASHTABLE_SIZE', None),
    140: ('REDUCE_ALL', 'AddReduceAll'),
    141: ('CONV_3D_TRANSPOSE', None),
    142: ('VAR_HANDLE', 'AddVarHandle'),
    143: ('READ_VARIABLE', 'AddReadVariable'),
    144: ('ASSIGN_VARIABLE', 'AddAssignVariable'),
    145: ('BROADCAST_ARGS', 'AddBroadcastArgs'),
    146: ('RANDOM_STANDARD_NORMAL', None),
    147: ('BUCKETIZE', None),
    148: ('RANDOM_UNIFORM', None),
    149: ('MULTINOMIAL', None),
    150: ('GELU', None),
    151: ('DYNAMIC_UPDATE_SLICE', None),
    152: ('RELU_0_TO_1', None),
    153: ('UNSORTED_SEGMENT_PROD', None),
    154: ('UNSORTED_SEGMENT_MAX', None),
    155: ('UNSORTED_SEGMENT_SUM', None),
    156: ('ATAN2', None),
    157: ('UNSORTED_SEGMENT_MIN', None),
    158: ('SIGN', None),
    159: ('BITCAST', None),
    160: ('BITWISE_XOR', None),
    161: ('RIGHT_SHIFT', None),
}

# Custom (non-builtin) operators that the FSP TFLM ships a kernel source for,
# mapped to the MicroMutableOpResolver helper that registers them.
#
# Custom operators are identified by the name string embedded in the model's
# operator_codes table rather than by a builtin enum value, so they cannot be
# resolved through TFLITE_BUILTIN_OP_MAP. Each helper below calls AddCustom()
# with exactly the name string used as the key here, so a model whose custom op
# matches a key can be registered automatically.
#
# Every Register_* symbol these helpers reference is declared by
# micro_mutable_op_resolver.h itself, so emitting the call needs no extra
# include in the generated source.
#
# 'ethos-u' is intentionally absent: NPU builds register it from the
# use_npu code path, and listing it here would register it twice. The
# resolver also declares TFLM_DECODE and the Signal* helpers, but FSP ships
# no kernel source for them, so registering them would fail to link.
#
# An FSP/TFLM upgrade can invalidate this table. Verify resolver helpers
# and enabled kernel sources against the target project's TFLM release.
TFLM_CUSTOM_OP_MAP = {
    'TFLite_Detection_PostProcess': 'AddDetectionPostprocess',
    'CIRCULAR_BUFFER': 'AddCircularBuffer',
}

# TFLite tensor type enum → C type name and numpy dtype
TFLITE_DTYPE_MAP = {
    0: ('float', 'float32', 4),    # FLOAT32
    1: ('uint16_t', 'float16', 2), # FLOAT16 raw IEEE 754 binary16 storage
    2: ('int32_t', 'int32', 4),    # INT32
    3: ('uint8_t', 'uint8', 1),    # UINT8
    4: ('int64_t', 'int64', 8),    # INT64
    6: ('bool', 'bool', 1),        # BOOL
    7: ('int16_t', 'int16', 2),    # INT16
    9: ('int8_t', 'int8', 1),      # INT8
    10: ('double', 'float64', 8),  # FLOAT64
}

TFLITE_DTYPE_NAMES = {
    0: 'FLOAT32',
    1: 'FLOAT16',
    2: 'INT32',
    3: 'UINT8',
    4: 'INT64',
    5: 'STRING',
    6: 'BOOL',
    7: 'INT16',
    8: 'COMPLEX64',
    9: 'INT8',
    10: 'FLOAT64',
    11: 'COMPLEX128',
    12: 'UINT64',
    13: 'RESOURCE',
    14: 'VARIANT',
    15: 'UINT32',
    16: 'UINT16',
    17: 'INT4',
    18: 'BFLOAT16',
}


def _peak_concurrent_liveness(interp, tensor_bytes: dict) -> Optional[int]:
    """
    Compute peak concurrent arena bytes via per-operator liveness analysis.

    TFLM's memory planner reuses buffers for tensors whose lifetimes do not
    overlap, so the arena only has to hold the largest *simultaneously live*
    set — not the sum of all tensors, and not simply the two largest ones.

    A tensor is live from the operator that first references it until the
    operator that last references it. Constant tensors (weights/biases) are
    excluded: they are read directly out of the model flatbuffer and never
    occupy arena space.

    Returns None when the interpreter does not expose an operator schedule,
    so the caller can fall back to a coarser approximation.
    """
    try:
        ops_details = interp._get_ops_details()
    except Exception:
        return None
    if not ops_details:
        return None

    produced = set()
    for op in ops_details:
        for t in op.get('outputs', []):
            if t >= 0:
                produced.add(int(t))

    model_inputs = {int(d['index']) for d in interp.get_input_details()}
    model_outputs = {int(d['index']) for d in interp.get_output_details()}

    # Anything neither produced by an operator nor supplied as a model input is
    # a constant (weights, biases, reshape-shape vectors) and lives in flash.
    def in_arena(idx: int) -> bool:
        return idx in produced or idx in model_inputs

    first_use, last_use = {}, {}
    for step, op in enumerate(ops_details):
        for t in list(op.get('inputs', [])) + list(op.get('outputs', [])):
            t = int(t)
            if t < 0 or not in_arena(t):
                continue
            first_use.setdefault(t, step)
            last_use[t] = step

    # Model inputs must already be resident before the first operator runs, and
    # model outputs must survive until after the last one.
    last_step = len(ops_details) - 1
    for t in model_inputs:
        if t in tensor_bytes:
            first_use[t] = 0
            last_use.setdefault(t, 0)
    for t in model_outputs:
        if t in tensor_bytes:
            first_use.setdefault(t, last_step)
            last_use[t] = last_step

    if not first_use:
        return None

    peak = 0
    for step in range(len(ops_details)):
        live = sum(tensor_bytes.get(t, 0)
                   for t, start in first_use.items()
                   if start <= step <= last_use.get(t, start))
        peak = max(peak, live)

    return peak if peak > 0 else None


def _estimate_arena_size(model_path: Path, file_size: int) -> int:
    """
    Estimate TFLM tensor arena size using the TFLite interpreter.

    This is an *approximation*. The TFLM arena contains:
      1. Tensor buffers — the memory planner reuses buffers for tensors
         with non-overlapping lifetimes, so only the peak concurrent
         set matters (not the sum of all tensors).
      2. Operator scratch buffers — allocated per-op during Prepare().
         These are hardware-dependent (CMSIS-NN allocates differently
         than reference kernels or Xtensa nnlib) and CANNOT be predicted
         from Python.

    (1) is derived from a per-operator liveness analysis of the model's
    execution schedule; (2) is covered by the headroom multiplier below.

    Falls back to a heuristic for Vela-compiled models with custom ops.
    """
    try:
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflri
                interp = tflri.Interpreter(model_path=str(model_path))
            except ImportError:
                raise ImportError("no interpreter available")

        interp.allocate_tensors()

        # Collect byte sizes of every tensor (inputs, outputs, intermediates)
        tensor_sizes = []
        tensor_bytes = {}
        for t in interp.get_tensor_details():
            # np.dtype(...) resolves the tensor's element type to a concrete
            # dtype instance, whose .itemsize gives the byte width of one
            # element (e.g. 1 for INT8, 4 for FLOAT32).
            size = int(np.dtype(t['dtype']).itemsize)
            for dim in t['shape']:
                size *= int(dim)
            tensor_sizes.append(size)
            tensor_bytes[t['index']] = size

        peak_concurrent = _peak_concurrent_liveness(interp, tensor_bytes)
        estimator = "liveness"

        if peak_concurrent is None:
            # Fallback when the interpreter exposes no op schedule: approximate
            # the peak with the two largest tensors. This is known to be
            # inaccurate in both directions, so it is only a last resort.
            tensor_sizes.sort(reverse=True)
            peak_concurrent = (sum(tensor_sizes[:2]) if len(tensor_sizes) >= 2
                               else sum(tensor_sizes))
            estimator = "two-largest fallback"

        # Add 50% headroom for operator scratch buffers and TFLM internals.
        # This is deliberately generous because scratch allocations are
        # hardware-dependent (e.g. CMSIS-NN Conv2D pre-computes im2col
        # buffers, kernel_sums for MVE, etc.) and cannot be known here.
        num_tensors = len(tensor_sizes)
        bookkeeping = num_tensors * 64  # TfLiteTensor structs + quant params
        arena = int(peak_concurrent * 1.5) + bookkeeping
        arena = max(arena, 32 * 1024)          # floor at 32 KB
        arena = (arena + 15) & ~15              # align to 16 bytes (cache-line)

        print(f"  Arena size (estimated): {arena} bytes "
              f"(peak concurrent {peak_concurrent} [{estimator}] + 50% headroom "
              f"+ {bookkeeping} bookkeeping)")
        return arena

    except Exception:
        # Fallback heuristic for models the host interpreter can't load
        # (e.g. Vela-compiled .tflite with ethos-u custom op)
        arena = max(file_size * 2, 32 * 1024)
        arena = (arena + 15) & ~15              # align to 16 bytes (cache-line)
        print(f"  Arena size (heuristic fallback): {arena} bytes")
        return arena


def parse_tflite_model(model_path: Path, skip_arena_estimate: bool = False) -> Dict[str, Any]:
    """
    Parse a .tflite flatbuffer to extract model metadata:
    - operator codes (for MicroMutableOpResolver)
    - input/output tensor info (names, shapes, dtypes)
    
    skip_arena_estimate: set True for Vela-compiled NPU models where
    arena size is known from Vela output (avoids misleading fallback print).
    
    Returns dict with keys: 'ops', 'inputs', 'outputs', 'arena_size_estimate'
    """
    data = open(model_path, 'rb').read()
    buf = bytearray(data)
    
    def get_field_offset(table_offset, field_idx):
        vt_off = table_offset - struct.unpack_from('<i', buf, table_offset)[0]
        vt_size = struct.unpack_from('<H', buf, vt_off)[0]
        if (field_idx + 2) * 2 >= vt_size:
            return 0
        field_rel = struct.unpack_from('<H', buf, vt_off + (field_idx + 2) * 2)[0]
        if field_rel == 0:
            return 0
        return table_offset + field_rel
    
    def read_vector(offset):
        vec_offset = offset + struct.unpack_from('<I', buf, offset)[0]
        length = struct.unpack_from('<I', buf, vec_offset)[0]
        return vec_offset + 4, length
    
    def read_string(offset):
        str_offset = offset + struct.unpack_from('<I', buf, offset)[0]
        str_len = struct.unpack_from('<I', buf, str_offset)[0]
        return buf[str_offset + 4:str_offset + 4 + str_len].decode('utf-8')
    
    # Root table
    root_offset = struct.unpack_from('<I', buf, 0)[0]
    
    # --- Parse operator_codes (field 1) ---
    oc_field = get_field_offset(root_offset, 1)
    op_codes = []
    if oc_field:
        oc_vec_start, oc_count = read_vector(oc_field)
        for i in range(oc_count):
            elem_offset = oc_vec_start + i * 4
            table_offset = elem_offset + struct.unpack_from('<I', buf, elem_offset)[0]
            
            builtin_field = get_field_offset(table_offset, 3)
            if builtin_field:
                builtin_code = struct.unpack_from('<i', buf, builtin_field)[0]
            else:
                dep_field = get_field_offset(table_offset, 0)
                builtin_code = struct.unpack_from('<B', buf, dep_field)[0] if dep_field else 0
            
            # Check for custom op string (field 1)
            custom_field = get_field_offset(table_offset, 1)
            custom_name = None
            if custom_field:
                try:
                    custom_name = read_string(custom_field)
                except Exception:
                    pass
            
            op_codes.append((builtin_code, custom_name))
    
    # --- Parse subgraphs (field 2) → first subgraph's inputs/outputs/tensors ---
    sg_field = get_field_offset(root_offset, 2)
    inputs_info = []
    outputs_info = []
    tensors = []
    
    if sg_field:
        sg_vec_start, _ = read_vector(sg_field)
        # First subgraph
        sg_offset = sg_vec_start + struct.unpack_from('<I', buf, sg_vec_start)[0]
        
        # Subgraph fields: tensors(0), inputs(1), outputs(2), operators(3), name(4)
        tensors_field = get_field_offset(sg_offset, 0)
        inputs_field = get_field_offset(sg_offset, 1)
        outputs_field = get_field_offset(sg_offset, 2)
        
        # Parse tensors table
        tensors = []
        if tensors_field:
            t_vec_start, t_count = read_vector(tensors_field)
            for i in range(t_count):
                t_elem = t_vec_start + i * 4
                t_offset = t_elem + struct.unpack_from('<I', buf, t_elem)[0]
                
                # Tensor fields: shape(0), type(1), buffer(2), name(3), quantization(4)
                shape_field = get_field_offset(t_offset, 0)
                type_field = get_field_offset(t_offset, 1)
                name_field = get_field_offset(t_offset, 3)
                quant_field = get_field_offset(t_offset, 4)
                
                shape = []
                if shape_field:
                    sh_vec_start, sh_count = read_vector(shape_field)
                    for j in range(sh_count):
                        shape.append(struct.unpack_from('<i', buf, sh_vec_start + j * 4)[0])
                
                dtype = struct.unpack_from('<B', buf, type_field)[0] if type_field else 0
                name = ''
                if name_field:
                    try:
                        name = read_string(name_field)
                    except Exception:
                        name = f'tensor_{i}'
                
                # Parse quantization parameters (scale and zero_point)
                scale = 0.0
                zero_point = 0
                if quant_field:
                    try:
                        q_offset = quant_field + struct.unpack_from('<I', buf, quant_field)[0]
                        # QuantizationParameters fields: min(0), max(1), scale(2), zero_point(3)
                        scale_field = get_field_offset(q_offset, 2)
                        zp_field = get_field_offset(q_offset, 3)
                        if scale_field:
                            sc_vec_start, sc_count = read_vector(scale_field)
                            if sc_count > 0:
                                scale = struct.unpack_from('<f', buf, sc_vec_start)[0]
                        if zp_field:
                            zp_vec_start, zp_count = read_vector(zp_field)
                            if zp_count > 0:
                                zero_point = struct.unpack_from('<i', buf, zp_vec_start)[0]
                    except Exception:
                        pass
                
                tensors.append({'name': name, 'shape': shape, 'dtype': dtype, 'scale': scale, 'zero_point': zero_point})
        
        # Parse input indices
        if inputs_field:
            inp_vec_start, inp_count = read_vector(inputs_field)
            for i in range(inp_count):
                idx = struct.unpack_from('<i', buf, inp_vec_start + i * 4)[0]
                if 0 <= idx < len(tensors):
                    inputs_info.append(tensors[idx])
        
        # Parse output indices
        if outputs_field:
            out_vec_start, out_count = read_vector(outputs_field)
            for i in range(out_count):
                idx = struct.unpack_from('<i', buf, out_vec_start + i * 4)[0]
                if 0 <= idx < len(tensors):
                    outputs_info.append(tensors[idx])
    
    # Unique builtin ops used
    unique_ops = sorted(set(code for code, custom in op_codes if custom is None))
    custom_ops = [custom for code, custom in op_codes if custom is not None]
    
    # Arena size: use TFLite interpreter to get exact allocation if possible
    file_size = model_path.stat().st_size
    arena_estimate = 0 if skip_arena_estimate else _estimate_arena_size(model_path, file_size)
    
    return {
        'ops': unique_ops,
        'custom_ops': custom_ops,
        'inputs': inputs_info,
        'outputs': outputs_info,
        'arena_size_estimate': arena_estimate,
        'file_size': file_size,
        'num_tensors': len(tensors),
    }


# =============================================================================
# TFLM Backend - Vela NPU Compilation
# =============================================================================

def _vela_launcher() -> List[str]:
    """
    Build the argv prefix used to invoke Vela.

    Prefers the ``vela`` console script when it is on PATH, but falls back to
    ``<this interpreter> -m ethosu.vela`` when it is not. The fallback matters
    because ``ethos-u-vela`` is frequently installed into a virtual environment
    whose ``bin``/``Scripts`` directory is not on PATH — for example when the
    compiler is invoked as ``/path/to/venv/bin/python mcu_compile_tflm.py``
    without activating that environment. Without the fallback that setup fails
    with a misleading "not installed" error even though Vela is importable.
    """
    exe = shutil.which('vela')
    if exe:
        return [exe]
    return [sys.executable, '-m', 'ethosu.vela']


def _parse_vela_cpu_op_count(vela_output: str) -> int:
    """
    Return how many operators Vela left on the CPU.

    Vela prints a partition summary such as ``CPU operators = 1 (2.6%)``.
    A non-zero count means the model is only partially offloaded, so the
    reported SRAM figure describes the NPU partition alone.
    """
    import re

    match = re.search(r'CPU operators\s*=\s*(\d+)', vela_output)
    return int(match.group(1)) if match else 0


def _parse_vela_sram_bytes(vela_output: str) -> Optional[int]:
    """Parse Vela's total SRAM summary and return bytes, or None if absent."""
    import re

    match = re.search(
        r'Total\s+SRAM\s+used\s*:?\s+([\d,.]+)\s+([A-Za-z]+)',
        vela_output or '',
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1).replace(',', ''))
    unit = match.group(2).upper()
    multipliers = {
        'B': 1,
        'BYTES': 1,
        'KB': 1024,
        'KIB': 1024,
        'MB': 1024 * 1024,
        'MIB': 1024 * 1024,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return int(value * multiplier)

# TFLM arena bookkeeping that Vela's SRAM figure does not include. Every tensor
# gets a TfLiteEvalTensor plus planner alignment padding; every model I/O tensor
# additionally gets a persistent TfLiteTensor with its quantization params.
# Fitted to rnnoise_INT8 (12 tensors, 9 I/O): Vela 962 B, measured need 1992 B.
_TFLM_NPU_BASE_OVERHEAD = 512
_TFLM_NPU_PER_TENSOR_OVERHEAD = 16
_TFLM_NPU_PER_IO_TENSOR_OVERHEAD = 64


def _tflm_npu_arena_overhead(model_info: Dict[str, Any]) -> int:
    """Bytes to add on top of Vela's SRAM figure for TFLM internals."""
    num_io = len(model_info['inputs']) + len(model_info['outputs'])
    return (_TFLM_NPU_BASE_OVERHEAD
            + _TFLM_NPU_PER_TENSOR_OVERHEAD * model_info.get('num_tensors', 0)
            + _TFLM_NPU_PER_IO_TENSOR_OVERHEAD * num_io)


def run_vela(model_path: Path, output_dir: Path, vela_config: Dict[str, Any]) -> Tuple[Path, Optional[int]]:
    """
    Run ARM Vela compiler on a .tflite to produce NPU-optimized .tflite.
    
    Returns (path_to_vela_tflite, sram_bytes) where sram_bytes is the
    'Total SRAM used' reported by Vela — the exact arena size needed
    for the Ethos-U55 command stream and I/O buffers.
    Returns sram_bytes=None if it cannot be parsed from Vela output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove a previous run's output so a Vela run that writes nothing cannot
    # silently reuse it.
    (output_dir / f"{model_path.stem}_vela.tflite").unlink(missing_ok=True)

    memory_mode = vela_config.get('memory_mode', 'Sram_Only')
    if vela_config.get('enable_ospi', False):
        # External memory on RA8P1 uses Shared_Sram.
        memory_mode = 'Shared_Sram'

    optimization = vela_config.get('optimise')
    if optimization is None:
        optimization = DEFAULT_VELA_OPTIMIZATION
    
    cmd = [
        *_vela_launcher(),
        str(model_path),
        '--output-dir', str(output_dir),
        '--accelerator-config', vela_config.get('accel_config', 'ethos-u55-256'),
        '--system-config', vela_config.get('sys_config', 'RA8P1'),
        '--memory-mode', memory_mode,
        '--optimise', optimization,
    ]
    
    config_file = vela_config.get('config_file')
    tmp_ini_fd = None
    if config_file:
        cmd.extend(['--config', config_file])
    else:
        # Write embedded config to an auto-cleaned temp file
        import tempfile
        tmp_ini_fd = tempfile.NamedTemporaryFile(
            mode='w', suffix='.ini', prefix='vela_ra8p1_', delete=False
        )
        tmp_ini_fd.write(VELA_RA8P1_CONFIG)
        tmp_ini_fd.close()
        cmd.extend(['--config', tmp_ini_fd.name])
    
    print(f"  Running Vela: {' '.join(cmd)}")
    
    vela_result = None
    try:
        vela_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if vela_result.stdout:
            print(f"  Vela output: {vela_result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Vela compilation failed:\n{e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "Vela not found. Install it into the interpreter running this "
            "script with: python -m pip install ethos-u-vela"
        )
    finally:
        # Clean up temp config file
        if tmp_ini_fd is not None:
            try:
                os.unlink(tmp_ini_fd.name)
            except OSError:
                pass

    # Save full Vela report for offline analysis
    vela_report_path = output_dir / "vela_report.txt"
    vela_report_path.write_text(vela_result.stdout or "", encoding='utf-8')
    print(f"  Vela report saved: {vela_report_path}")

    # Parse 'Total SRAM used' from Vela summary — this is the exact arena size
    # needed on-device for the NPU command stream + I/O buffers.
    vela_sram_bytes = None
    if vela_result.stdout:
        cpu_ops = _parse_vela_cpu_op_count(vela_result.stdout)
        vela_sram_bytes = _parse_vela_sram_bytes(vela_result.stdout)
        if vela_sram_bytes is None:
            print("  Warning: Could not parse 'Total SRAM used' from Vela output — "
                  "this is expected when Vela leaves the entire model on CPU. "
                  "Arena size will be estimated from the Vela output model. "
                  "Check vela_report.txt for the actual SRAM figure.")
        elif cpu_ops:
            # Vela only accounts for the operators it accelerates. Any operator
            # it leaves on the CPU still needs its own tensor buffers in the
            # TFLM arena, and those bytes appear nowhere in the Vela summary.
            # Trusting the summary for a partially-offloaded model under-sizes
            # the arena, so AllocateTensors() fails and ModelInit() returns -2.
            # Fall back to the host-side estimate, which covers the whole graph.
            print(f"  Vela left {cpu_ops} operator(s) on the CPU; its SRAM figure "
                  f"({vela_sram_bytes} bytes) covers only the NPU partition. "
                  "Falling back to the host-side arena estimate for the full graph.")
            vela_sram_bytes = None
        else:
            # Vela's "Total SRAM used" covers the NPU feature maps and scratch
            # only. TFLM bookkeeping is added by the caller via
            # _tflm_npu_arena_overhead(), once the tensor count is known.
            vela_sram_bytes = int(vela_sram_bytes)
            print(f"  Vela SRAM requirement: {vela_sram_bytes} bytes (from Vela summary)")
    
    # Vela always writes <model_stem>_vela.tflite. Never pick another
    # *_vela.tflite: the folder is reused across runs, so any other match
    # would be a stale file from an earlier compile.
    vela_output = output_dir / f"{model_path.stem}_vela.tflite"
    if not vela_output.exists():
        raise RuntimeError(
            f"Vela finished but did not write the expected output {vela_output}"
        )

    return vela_output, vela_sram_bytes


# =============================================================================
# TFLM Backend - Quantization via TensorFlow Lite Calibrator
# =============================================================================

def run_tflite_quantization_quality_check(
    float_model_content: bytes,
    quantized_model_content: bytes,
    representative_dataset_fn,
    output_dir: Path,
    using_random_data: bool,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """
    Run TFLite QuantizationDebugger after quantization and emit warning-only quality checks.

    This function never raises; on any failure it prints a warning and returns.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("  Warning: TensorFlow not available, skipping quantization quality checkpoint")
        return

    q_thresholds = {
        'cosine_similarity': 0.997,
        'psnr_db': 30.0,
    }
    if thresholds:
        q_thresholds.update(thresholds)

    output_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_outputs(outputs) -> np.ndarray:
        if outputs is None:
            return np.array([], dtype=np.float32)
        chunks = [np.asarray(o, dtype=np.float32).ravel() for o in outputs]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    def _paired_vectors(float_outputs, quant_outputs) -> Tuple[np.ndarray, np.ndarray]:
        ref = _flatten_outputs(float_outputs)
        got = _flatten_outputs(quant_outputs)
        if ref.size == 0 or got.size == 0:
            return ref, got
        if ref.size != got.size:
            n = min(ref.size, got.size)
            return ref[:n], got[:n]
        return ref, got

    def _mse(float_outputs, quant_outputs) -> float:
        ref, got = _paired_vectors(float_outputs, quant_outputs)
        if ref.size == 0 or got.size == 0:
            return float('inf')
        diff = ref - got
        return float(np.mean(diff ** 2))

    def _max_abs(float_outputs, quant_outputs) -> float:
        ref, got = _paired_vectors(float_outputs, quant_outputs)
        if ref.size == 0 or got.size == 0:
            return float('inf')
        return float(np.max(np.abs(ref - got)))

    def _cosine_similarity(float_outputs, quant_outputs) -> float:
        ref, got = _paired_vectors(float_outputs, quant_outputs)
        if ref.size == 0 or got.size == 0:
            return 0.0
        n_ref = float(np.linalg.norm(ref))
        n_got = float(np.linalg.norm(got))
        if n_ref == 0.0 and n_got == 0.0:
            return 1.0
        if n_ref == 0.0 or n_got == 0.0:
            return 0.0
        cos = float(np.dot(ref, got) / (n_ref * n_got))
        return max(-1.0, min(1.0, cos))

    def _psnr_db(float_outputs, quant_outputs) -> float:
        ref, got = _paired_vectors(float_outputs, quant_outputs)
        if ref.size == 0 or got.size == 0:
            return 0.0
        diff = ref - got
        mse = float(np.mean(diff ** 2))
        if mse == 0.0:
            return float('inf')
        peak = float(np.max(np.abs(ref))) if ref.size else 1.0
        if peak == 0.0:
            peak = 1.0
        return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

    def _reshape_to_detail(arr: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
        target_shape = [int(d) if int(d) > 0 else 1 for d in detail['shape']]
        if list(arr.shape) != target_shape:
            arr = arr.reshape(target_shape)
        return arr

    def _quantize_input_for_detail(arr: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
        target_dtype = np.dtype(detail['dtype'])
        if np.issubdtype(target_dtype, np.integer):
            scale, zero_point = detail.get('quantization', (0.0, 0))
            if scale is not None and scale != 0:
                q = np.round(arr.astype(np.float32) / float(scale) + float(zero_point))
                info = np.iinfo(target_dtype)
                q = np.clip(q, info.min, info.max)
                return q.astype(target_dtype)
            return arr.astype(target_dtype)
        return arr.astype(target_dtype)

    def _dequantize_output_for_detail(arr: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.integer):
            scale, zero_point = detail.get('quantization', (0.0, 0))
            if scale is not None and scale != 0:
                return (arr.astype(np.float32) - float(zero_point)) * float(scale)
            return arr.astype(np.float32)
        return arr.astype(np.float32)

    layer_stats_path = output_dir / 'layer_stats.csv'
    debugger_error = None
    try:
        debug_options = tf.lite.experimental.QuantizationDebugOptions(
            model_debug_metrics={
                'mse': _mse,
                'max_abs_error': _max_abs,
                'cosine_similarity': _cosine_similarity,
                'psnr_db': _psnr_db,
            }
        )
        debugger = tf.lite.experimental.QuantizationDebugger(
            quant_debug_model_content=quantized_model_content,
            float_model_content=float_model_content,
            debug_dataset=representative_dataset_fn,
            debug_options=debug_options,
        )
        debugger.run()
        with open(layer_stats_path, 'w', encoding='utf-8') as f:
            debugger.layer_statistics_dump(f)
    except Exception as e:
        debugger_error = str(e)

    try:
        float_interp = tf.lite.Interpreter(model_content=float_model_content)
        quant_interp = tf.lite.Interpreter(model_content=quantized_model_content)
        float_interp.allocate_tensors()
        quant_interp.allocate_tensors()

        float_inputs = float_interp.get_input_details()
        quant_inputs = quant_interp.get_input_details()
        float_outputs = float_interp.get_output_details()
        quant_outputs = quant_interp.get_output_details()

        sample_count = 0
        mse_vals = []
        max_abs_vals = []
        cos_vals = []
        psnr_vals = []

        for sample in representative_dataset_fn():
            sample_list = list(sample) if isinstance(sample, (list, tuple)) else [sample]
            if len(sample_list) < len(float_inputs):
                continue

            for i, detail in enumerate(float_inputs):
                arr = np.asarray(sample_list[i])
                arr = _reshape_to_detail(arr, detail).astype(np.dtype(detail['dtype']))
                float_interp.set_tensor(detail['index'], arr)

            for i, detail in enumerate(quant_inputs):
                arr = np.asarray(sample_list[i])
                arr = _reshape_to_detail(arr, detail)
                arr = _quantize_input_for_detail(arr, detail)
                quant_interp.set_tensor(detail['index'], arr)

            float_interp.invoke()
            quant_interp.invoke()

            ref_outs = [float_interp.get_tensor(d['index']) for d in float_outputs]
            got_outs = [_dequantize_output_for_detail(quant_interp.get_tensor(d['index']), d) for d in quant_outputs]

            mse_vals.append(_mse(ref_outs, got_outs))
            max_abs_vals.append(_max_abs(ref_outs, got_outs))
            cos_vals.append(_cosine_similarity(ref_outs, got_outs))
            psnr_vals.append(_psnr_db(ref_outs, got_outs))
            sample_count += 1

        if sample_count == 0:
            raise RuntimeError("No samples available for quality checkpoint")

        model_stats = {
            'samples_evaluated': sample_count,
            'mse': float(np.mean(mse_vals)),
            'max_abs_error': float(np.max(max_abs_vals)),
            'cosine_similarity': float(np.mean(cos_vals)),
            'psnr_db': float(np.mean(psnr_vals)),
            'using_random_calibration_data': using_random_data,
            'thresholds': q_thresholds,
            'debugger_layer_stats_csv': str(layer_stats_path) if layer_stats_path.exists() else '',
            'debugger_error': debugger_error or '',
        }

        low_cos = model_stats['cosine_similarity'] < q_thresholds['cosine_similarity']
        low_psnr = model_stats['psnr_db'] < q_thresholds['psnr_db']
        model_stats['warn_low_quality'] = bool(low_cos or low_psnr)

        summary_path = output_dir / 'model_stats.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(model_stats, f, indent=2)

        if layer_stats_path.exists():
            print(f"  Quantization debug report: {layer_stats_path}")
        elif debugger_error:
            print(f"  Note: Layer debug report unavailable: {debugger_error}")

        print(f"  Quantization debug summary: {summary_path}")
        print(
            "  Quantization quality checkpoint: "
            f"Cosine={model_stats['cosine_similarity']:.4f}, "
            f"PSNR={model_stats['psnr_db']:.2f} dB"
        )

        if using_random_data:
            print("  Note: Quality checkpoint used random calibration data")

        if model_stats['warn_low_quality']:
            print_warning(
                "  ⚠️  WARNING: Quantization quality checkpoint is below thresholds "
                f"(cosine<{q_thresholds['cosine_similarity']} or psnr<{q_thresholds['psnr_db']} dB)."
            )

    except Exception as e:
        print(f"  Warning: Quantization quality checkpoint failed: {e}")

def _split_calibration_array(
    data: np.ndarray,
    input_shape: List[int],
    num_samples: int,
    source: Path,
) -> List[np.ndarray]:
    """
    Split one numeric .npy array into samples using the model's input shape.

    The array is one sample when it matches the input shape, with or without its
    leading batch dimension of 1; otherwise its first axis must count samples.
    """
    shape = tuple(int(d) for d in input_shape)
    per_sample = {shape}
    if len(shape) > 1 and shape[0] == 1:
        per_sample.add(shape[1:])

    if data.shape in per_sample:
        return [data]
    if data.ndim >= 1 and data.shape[1:] in per_sample:
        available = int(data.shape[0])
        if 0 < available < num_samples:
            print(f"  Warning: {source.name} holds {available} calibration sample(s); "
                  f"--calib-num {num_samples} requested, using {available}")
        return [data[i] for i in range(min(num_samples, available))]

    expected = ' or '.join(str(s) for s in sorted(per_sample, key=len))
    raise ValueError(
        f"Calibration data {source} has shape {data.shape}; expected one sample "
        f"shaped {expected}, or N samples stacked along a new first axis"
    )


def _load_calibration_npy(path: Path, allow_pickle: bool):
    try:
        return np.load(str(path), allow_pickle=allow_pickle)
    except ValueError as exc:
        if allow_pickle:
            raise
        raise ValueError(
            f"Calibration data {path} contains Python objects; a single-input model "
            "needs a numeric array"
        ) from exc


def _load_tflm_calibration_samples(
    calib_data_path: Optional[Path],
    num_samples: int,
    input_shape: Optional[List[int]] = None,
) -> Optional[List]:
    """
    Load explicit TFLM calibration samples from a .npy file or directory.

    input_shape: the input shape of a single-input model. When given, a .npy
    file is split with it (see _split_calibration_array) and pickled objects
    are refused, since only multi-input models need object arrays.
    """
    if num_samples < 1:
        raise ValueError("Calibration sample count (--calib-num) must be at least 1")

    if calib_data_path is None:
        return None

    calib_data_path = Path(calib_data_path)
    if not calib_data_path.exists():
        raise ValueError(f"Calibration data path does not exist: {calib_data_path}")

    allow_pickle = input_shape is None

    if calib_data_path.is_dir():
        npy_files = sorted(calib_data_path.glob("*.npy"))
        if not npy_files:
            raise ValueError(
                f"Calibration data directory contains no .npy files: {calib_data_path}"
            )
        return [
            _load_calibration_npy(sample_path, allow_pickle)
            for sample_path in npy_files[:num_samples]
        ]

    if not calib_data_path.is_file():
        raise ValueError(f"Calibration data path is not a file or directory: {calib_data_path}")
    if calib_data_path.suffix.lower() != '.npy':
        raise ValueError(f"Calibration data file must use the .npy extension: {calib_data_path}")

    data = _load_calibration_npy(calib_data_path, allow_pickle)
    if isinstance(data, np.ndarray) and data.dtype == object:
        samples = list(data)[:num_samples]
    elif input_shape is not None:
        samples = _split_calibration_array(data, input_shape, num_samples, calib_data_path)
    elif isinstance(data, np.ndarray) and data.ndim > 0 and int(data.shape[0]) >= num_samples:
        samples = [data[i] for i in range(min(num_samples, int(data.shape[0])))]
    else:
        samples = [data]

    if not samples:
        raise ValueError(f"Calibration data file contains no samples: {calib_data_path}")
    return samples


def quantize_tflite_model(
    model_path: Path,
    output_dir: Path,
    calib_data_path: Optional[Path] = None,
    num_samples: int = 5
) -> Path:
    """
    Quantize a FP32 .tflite model to INT8 using TensorFlow Lite calibrator.
    
    Returns path to quantized .tflite.
    """
    try:
        import tensorflow as tf
        from tensorflow.lite.python.optimize import calibrator
    except ImportError:
        raise RuntimeError(
            "TensorFlow Lite quantizer not available in this environment.\n"
            "Install TensorFlow or provide a pre-quantized INT8 .tflite model."
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}_int8.tflite"
    
    print("  Quantizing with TensorFlow Lite calibrator...")

    model_content = model_path.read_bytes()
    interp = tf.lite.Interpreter(model_content=model_content)
    interp.allocate_tensors()
    input_details = interp.get_input_details()

    def _normalize_shape(shape):
        return [int(d) if int(d) > 0 else 1 for d in shape]

    def _to_input_batch(sample):
        """Convert a sample into list of input tensors matching model signature."""
        if len(input_details) == 1:
            d = input_details[0]
            target_shape = _normalize_shape(d['shape'])
            target_dtype = d['dtype']
            arr = np.asarray(sample, dtype=target_dtype)
            if list(arr.shape) != target_shape:
                arr = arr.reshape(target_shape)
            return [arr]

        # Multi-input model: expect list/tuple aligned to inputs
        if not isinstance(sample, (list, tuple)) or len(sample) != len(input_details):
            raise RuntimeError(
                "Calibration sample format mismatch for multi-input model. "
                "Expected list/tuple matching model inputs."
            )

        batch = []
        for s, d in zip(sample, input_details):
            target_shape = _normalize_shape(d['shape'])
            target_dtype = d['dtype']
            arr = np.asarray(s, dtype=target_dtype)
            if list(arr.shape) != target_shape:
                arr = arr.reshape(target_shape)
            batch.append(arr)
        return batch

    single_input_shape = (
        _normalize_shape(input_details[0]['shape']) if len(input_details) == 1 else None
    )
    calib_samples = _load_tflm_calibration_samples(
        calib_data_path, num_samples, input_shape=single_input_shape)
    if calib_samples is not None:
        print(f"  Using calibration data: {calib_data_path} ({len(calib_samples)} samples)")
    else:
        print("  Warning: No calibration data provided, using random representative data")

    def representative_dataset():
        if calib_samples is not None:
            for sample in calib_samples:
                yield _to_input_batch(sample)
            return

        # Fallback representative samples
        for _ in range(max(1, num_samples)):
            batch = []
            for d in input_details:
                shape = _normalize_shape(d['shape'])
                dtype = d['dtype']
                if dtype == np.float32:
                    arr = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                else:
                    arr = np.zeros(shape, dtype=dtype)
                batch.append(arr)
            yield batch

    cal = calibrator.Calibrator(model_content)
    quantized_model = cal.calibrate_and_quantize(
        representative_dataset,
        input_type=tf.int8,
        output_type=tf.int8,
        allow_float=False,
        activations_type=tf.int8,
        bias_type=tf.int32,
        resize_input=True,
        disable_per_channel=False,
    )

    output_path.write_bytes(quantized_model)

    # Post-quantization quality checkpoint (warning-only).
    run_tflite_quantization_quality_check(
        float_model_content=model_content,
        quantized_model_content=quantized_model,
        representative_dataset_fn=representative_dataset,
        output_dir=output_dir / 'quant_debug',
        using_random_data=(calib_samples is None),
    )
    
    print(f"  Quantized model saved: {output_path}")
    return output_path


# =============================================================================
# TFLM Backend - C Code Generation
# =============================================================================

def _make_c_identifier(name: str) -> str:
    """Convert tensor name to valid C identifier."""
    if not name:
        return 'tensor'
    result = name.replace('/', '_').replace(':', '_').replace('.', '_').replace('-', '_')
    result = ''.join(c if c.isalnum() or c == '_' else '_' for c in result)
    if not result:
        return 'tensor'
    if result[0].isdigit():
        result = '_' + result
    return result


def _normalize_suffix(suffix: Optional[str]) -> str:
    """
    Return the suffix used for folder names, file names and C symbols.

    The caller always adds the '_' separator, so leading/trailing underscores
    are stripped: '--suffix a' and '--suffix _a' give the same result. A suffix
    never starts a C identifier, so a leading digit needs no '_' prefix.
    """
    if not suffix:
        return ''
    return ''.join(c if c.isalnum() else '_' for c in str(suffix).strip()).strip('_')


def _quantized_dtype_info(dtype: int) -> Tuple[str, str, str]:
    """Return C type and saturation limits for a quantized TFLite dtype."""
    dtype_info = {
        2: ('int32_t', 'INT32_MIN', 'INT32_MAX'),
        3: ('uint8_t', '0', '255'),
        7: ('int16_t', '-32768', '32767'),
        9: ('int8_t', '-128', '127'),
    }
    try:
        return dtype_info[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported quantized tensor dtype: {dtype}") from exc


def _format_c_float_literal(value: float) -> str:
    """Format a finite value as a valid single-precision C literal."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Cannot generate a C float literal for non-finite value: {value}")

    literal = f'{value:.10g}'
    if '.' not in literal and 'e' not in literal.lower():
        literal += '.0'
    return literal + 'f'


def _require_supported_builtin_ops(ops: List[int]) -> None:
    """
    Raise when the model uses builtin operators the FSP TFLM cannot run.

    Such a model would compile into a model.cc that never registers the
    operator, and the failure would only appear on the board as ModelInit()
    returning -2.
    """
    unsupported = []
    for op_code in ops:
        op_info = TFLITE_BUILTIN_OP_MAP.get(op_code)
        if not (op_info and op_info[1]):
            name = op_info[0] if op_info else 'UNKNOWN'
            unsupported.append(f"{name} ({op_code})")
    if unsupported:
        raise ValueError(
            "Model uses builtin operator(s) that the FSP TFLM library cannot run: "
            f"{', '.join(unsupported)}. Replace or remove them in the source model."
        )


def _io_tensor_c_type(tensor: Dict[str, Any], tensor_kind: str) -> str:
    """Return a safe generated C pointer type for a model input or output."""
    dtype = tensor.get('dtype')
    dtype_info = TFLITE_DTYPE_MAP.get(dtype)
    if dtype_info is None:
        dtype_name = TFLITE_DTYPE_NAMES.get(dtype, f'UNKNOWN({dtype})')
        tensor_name = tensor.get('name') or '<unnamed>'
        raise ValueError(
            f"Unsupported {tensor_kind} tensor dtype {dtype_name} for tensor "
            f"'{tensor_name}'. The TFLM backend cannot generate a safe C pointer "
            "accessor for this tensor type."
        )
    return dtype_info[0]


def _generate_model_data_files(model_path: Path, output_dir: Path, model_info: Dict, suffix: str = '') -> Tuple[str, str]:
    """
    Generate model_data.c and model_data.h - the .tflite as a C byte array.
    When suffix is non-empty, all symbols are namespaced to allow multi-model builds.
    """
    sfx = _normalize_suffix(suffix)
    SFX = sfx.upper()
    guard = f'MODEL_DATA_{SFX}_H' if sfx else 'MODEL_DATA_H'
    var_data = f'g_model_data_{sfx}' if sfx else 'g_model_data'
    var_size = f'g_model_data_size_{sfx}' if sfx else 'g_model_data_size'
    macro_size = f'MODEL_DATA_SIZE_{SFX}' if sfx else 'MODEL_DATA_SIZE'
    h_file = f'model_data_{sfx}.h' if sfx else 'model_data.h'
    c_file = f'model_data_{sfx}.c' if sfx else 'model_data.c'

    data = Path(model_path).read_bytes()
    model_size = len(data)
    
    # Generate hex array
    hex_lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        hex_lines.append(f'  {hex_str},')
    hex_content = '\n'.join(hex_lines)
    
    header = f"""\
// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). The model byte array in the
// corresponding .c file retains the original customer model's license.
#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stddef.h>

/* TFLite model flatbuffer data */
extern const uint8_t {var_data}[];
extern const size_t {var_size};

#define {macro_size} {model_size}

#endif /* {guard} */
"""
    
    source = f"""\
// SPDX-License-Identifier: BSD-3-Clause
// This wrapper file is BSD-3-Clause. The model byte array below is the
// customer's model data and is governed by the customer model's own
// original license, not by this wrapper's license.
#include "{h_file}"

/* TFLite model flatbuffer ({model_size} bytes) */
__attribute__((aligned(16)))
const uint8_t {var_data}[{model_size}] = {{
{hex_content}
}};

const size_t {var_size} = {model_size};
"""
    
    (output_dir / h_file).write_text(header, encoding='utf-8')
    (output_dir / c_file).write_text(source, encoding='utf-8')
    
    # Store generated names in model_info for _generate_model_files to use
    model_info['_sfx'] = sfx
    model_info['_data_header'] = h_file
    model_info['_var_data'] = var_data
    
    return header, source


def _generate_model_files(
    output_dir: Path,
    model_info: Dict,
    use_npu: bool,
    arena_size: int = None,
    suffix: str = ''
) -> None:
    """
    Generate model.h and model.cc with unified RunModel(bool) API using TFLM.
    
    The generated code uses MicroMutableOpResolver with only the ops needed.
    When suffix is non-empty, all public symbols are namespaced to allow
    multiple models to coexist in the same binary.
    """
    inputs = model_info['inputs']
    outputs = model_info['outputs']
    ops = model_info['ops']
    custom_ops = model_info.get('custom_ops', [])
    _require_supported_builtin_ops(ops)

    # Custom ops are keyed by name string and, unlike builtin ops, arrive
    # undeduplicated. AddCustom() rejects the same name twice, so collapse
    # duplicates here while preserving first-seen order. 'ethos-u' is skipped
    # because NPU builds register it from the use_npu path below.
    seen_custom = set()
    auto_custom_ops, unknown_custom_ops = [], []
    for op in custom_ops:
        if not op or op == 'ethos-u' or op in seen_custom:
            continue
        seen_custom.add(op)
        if op in TFLM_CUSTOM_OP_MAP:
            auto_custom_ops.append(op)
        else:
            unknown_custom_ops.append(op)

    if auto_custom_ops:
        print(
            f"  Custom op(s) auto-registered: {auto_custom_ops}"
        )
    if unknown_custom_ops:
        print(
            f"  WARNING: Model contains unknown custom op(s): {unknown_custom_ops}. "
            "TFLM ships no kernel for these, so they cannot be auto-registered in "
            "the MicroMutableOpResolver. You must supply the kernel and add a manual "
            "resolver.AddCustom(...) call in the generated .cc file (and widen the "
            "MicroMutableOpResolver<N> template argument by one for each), otherwise "
            "AllocateTensors() will return kTfLiteError at runtime."
        )

    sfx = _normalize_suffix(suffix)
    SFX = sfx.upper()
    guard = f'MODEL_{SFX}_H' if sfx else 'MODEL_H'
    fn_init = f'ModelInit_{sfx}' if sfx else 'ModelInit'
    fn_run = f'RunModel_{sfx}' if sfx else 'RunModel'
    fn_get_arena_size = f'GetTensorArenaSizeBytes_{sfx}' if sfx else 'GetTensorArenaSizeBytes'
    fn_get_arena_used = f'GetTensorArenaUsedBytes_{sfx}' if sfx else 'GetTensorArenaUsedBytes'
    fn_get_last_status = f'GetLastInvokeStatus_{sfx}' if sfx else 'GetLastInvokeStatus'
    arena_macro = f'TENSOR_ARENA_SIZE_{SFX}' if sfx else 'TENSOR_ARENA_SIZE'
    arena_var = f'tensor_arena_{sfx}' if sfx else 'tensor_arena'
    data_header = model_info.get('_data_header', 'model_data.h')
    var_data = model_info.get('_var_data', 'g_model_data')
    h_file = f'model_{sfx}.h' if sfx else 'model.h'
    cc_file = f'model_{sfx}.cc' if sfx else 'model.cc'
    inp_prefix = f'GetModelInputPtr_{sfx}_' if sfx else 'GetModelInputPtr_'
    out_prefix = f'GetModelOutputPtr_{sfx}_' if sfx else 'GetModelOutputPtr_'

    if arena_size is None:
        arena_size = model_info['arena_size_estimate']
    if arena_size == 0:
        print(
            "  WARNING: arena_size_estimate is 0 — Vela SRAM parse likely failed or "
            "skip_arena_estimate was set with no fallback value. "
            "You must manually set TENSOR_ARENA_SIZE in the generated .cc file before "
            "calling AllocateTensors()."
        )
    # The single place the 5% safety buffer is applied (covers TFLM internals
    # and hardware-dependent scratch buffers regardless of whether the size came
    # from the TFLite interpreter estimate, the heuristic fallback, or Vela).
    arena_size = int(arena_size * 1.05)
    # Align to 16 bytes (cache-line)
    arena_size = ((arena_size + 15) // 16) * 16
    print(f"  {arena_macro}: {arena_size} bytes (includes 5% safety buffer)")
    
    # Build op resolver entries
    # Use INT8-specialized kernels only when model I/O is fully INT8/UINT8.
    # Mixed-precision models (e.g. INT16 activations) must use generic kernels.
    io_dtypes = [t.get('dtype') for t in (inputs + outputs)]
    use_int8_specialized_kernels = bool(io_dtypes) and all(d in (3, 9) for d in io_dtypes)

    num_ops = len(ops) + len(auto_custom_ops) + (1 if use_npu else 0)  # +1 for ethos-u custom op
    op_resolver_adds = []
    for op_code in ops:
        op_info = TFLITE_BUILTIN_OP_MAP.get(op_code)
        if op_info and op_info[1]:
            # Some FSP/TFLM CPU builds expose INT8-specific kernel registrations
            # but do not export the generic Register_* symbols at link time.
            # Use explicit INT8 registrations for common INT8 ops to avoid
            # unresolved symbols like Register_FULLY_CONNECTED()/Register_SOFTMAX().
            if not use_npu and op_code == 9 and use_int8_specialized_kernels:  # FULLY_CONNECTED
                op_resolver_adds.append(
                    '  resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());'
                )
            elif not use_npu and op_code == 25 and use_int8_specialized_kernels:  # SOFTMAX
                op_resolver_adds.append(
                    '  resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());'
                )
            else:
                op_resolver_adds.append(f'  resolver.{op_info[1]}();')
        else:
            # Unreachable after _require_supported_builtin_ops() above.
            raise ValueError(f"Unsupported builtin operator code {op_code}")
    
    for custom_op in auto_custom_ops:
        op_resolver_adds.append(f'  resolver.{TFLM_CUSTOM_OP_MAP[custom_op]}();  // custom op "{custom_op}"')

    for unknown_op in unknown_custom_ops:
        op_resolver_adds.append(
            f'  // TODO: unregistered custom op "{unknown_op}" - supply its kernel, add a\n'
            f'  //       resolver.AddCustom("{unknown_op}", Register_...()); call here, and\n'
            f'  //       widen the MicroMutableOpResolver<> template argument by one.'
        )

    if use_npu:
        op_resolver_adds.append('  resolver.AddCustom(tflite::GetString_ETHOSU(), tflite::Register_ETHOSU());')
    
    op_resolver_block = '\n'.join(op_resolver_adds)
    
    # Build input/output pointer functions
    input_decls = []
    input_impls = []
    output_decls = []
    output_impls = []
    
    for i, inp in enumerate(inputs):
        c_name = _make_c_identifier(inp['name'])
        c_type = _io_tensor_c_type(inp, 'input')
        
        input_decls.append(f'{c_type}* {inp_prefix}{c_name}();')
        input_impls.append(f"""\
{c_type}* {inp_prefix}{c_name}() {{
  return (({c_type}*)model_input_tensor[{i}]->data.raw);
}}""")
    
    for i, out in enumerate(outputs):
        c_name = _make_c_identifier(out['name'])
        c_type = _io_tensor_c_type(out, 'output')
        
        output_decls.append(f'{c_type}* {out_prefix}{c_name}();')
        output_impls.append(f"""\
{c_type}* {out_prefix}{c_name}() {{
  return (({c_type}*)model_output_tensor[{i}]->data.raw);
}}""")
    
    input_decls_str = '\n'.join(input_decls)
    output_decls_str = '\n'.join(output_decls)
    input_impls_str = '\n\n'.join(input_impls)
    output_impls_str = '\n\n'.join(output_impls)

    # Build quantization constants and typed helpers for quantized I/O tensors.
    QUANTIZED_DTYPES = {2, 3, 7, 9}  # INT32, UINT8, INT16, INT8
    quant_lines = []
    quantized_inputs = [
        inp for inp in inputs
        if inp['dtype'] in QUANTIZED_DTYPES and inp.get('scale', 0.0) != 0.0
    ]
    quantized_outputs = [
        out for out in outputs
        if out['dtype'] in QUANTIZED_DTYPES and out.get('scale', 0.0) != 0.0
    ]

    # With a suffix, the macro names carry it as a prefix (PALM_INPUT_SCALE_...)
    # so headers from several models can be included in one translation unit.
    mp = f'{SFX}_' if sfx else ''

    for inp in quantized_inputs:
        c_name = _make_c_identifier(inp['name']).upper()
        quant_lines.append(f'#define {mp + "INPUT_SCALE_" + c_name:<{48 + len(mp)}s} {_format_c_float_literal(inp["scale"])}')
        quant_lines.append(f'#define {mp + "INPUT_ZERO_POINT_" + c_name:<{49 + len(mp)}s} ({inp["zero_point"]})')

    for out in quantized_outputs:
        c_name = _make_c_identifier(out['name']).upper()
        quant_lines.append(f'#define {mp + "OUTPUT_SCALE_" + c_name:<{48 + len(mp)}s} {_format_c_float_literal(out["scale"])}')
        quant_lines.append(f'#define {mp + "OUTPUT_ZERO_POINT_" + c_name:<{49 + len(mp)}s} ({out["zero_point"]})')

    if quantized_inputs or quantized_outputs:
        helper_lines = ['/* Typed per-tensor quantization helpers. */']
        helper_namespace = f'_{sfx}' if sfx else ''

        for inp in quantized_inputs:
            tensor_id = _make_c_identifier(inp['name'])
            macro_id = tensor_id.upper()
            c_type, q_min, q_max = _quantized_dtype_info(inp['dtype'])
            helper_name = f'model_quantize{helper_namespace}_input_{tensor_id}'
            helper_lines.extend([
                f'static inline {c_type} {helper_name}(float x) {{',
                f'    int64_t v = (int64_t)roundf(x / {mp}INPUT_SCALE_{macro_id}) + {mp}INPUT_ZERO_POINT_{macro_id};',
                f'    if (v < {q_min}) v = {q_min};',
                f'    if (v > {q_max}) v = {q_max};',
                f'    return ({c_type})v;',
                '}',
            ])

        for out in quantized_outputs:
            tensor_id = _make_c_identifier(out['name'])
            macro_id = tensor_id.upper()
            c_type, _, _ = _quantized_dtype_info(out['dtype'])
            helper_name = f'model_dequantize{helper_namespace}_output_{tensor_id}'
            helper_lines.extend([
                f'static inline float {helper_name}({c_type} x) {{',
                f'    return ((float)(x) - (float){mp}OUTPUT_ZERO_POINT_{macro_id}) * {mp}OUTPUT_SCALE_{macro_id};',
                '}',
            ])

        # Preserve the existing generic API when all quantized inputs or
        # outputs share one dtype. Mixed-dtype models use per-tensor helpers.
        input_dtypes = {inp['dtype'] for inp in quantized_inputs}
        if len(input_dtypes) == 1:
            c_type, q_min, q_max = _quantized_dtype_info(next(iter(input_dtypes)))
            legacy_name = f'model_quantize_{sfx}' if sfx else 'model_quantize'
            helper_lines.extend([
                '/* Backward-compatible helper for homogeneous quantized inputs. */',
                f'static inline {c_type} {legacy_name}(float x, float scale, int zero_point) {{',
                '    int64_t v = (int64_t)roundf(x / scale) + (int64_t)zero_point;',
                f'    if (v < {q_min}) v = {q_min};',
                f'    if (v >  {q_max}) v = {q_max};',
                f'    return ({c_type})v;',
                '}',
            ])

        output_dtypes = {out['dtype'] for out in quantized_outputs}
        if len(output_dtypes) == 1:
            c_type, _, _ = _quantized_dtype_info(next(iter(output_dtypes)))
            legacy_name = f'model_dequantize_{sfx}' if sfx else 'model_dequantize'
            helper_lines.extend([
                '/* Backward-compatible helper for homogeneous quantized outputs. */',
                f'static inline float {legacy_name}({c_type} x, float scale, int zero_point) {{',
                '    return ((float)(x) - (float)zero_point) * scale;',
                '}',
            ])

        quant_constants_str = (
            '/* Quantization parameters (scale / zero_point) for each quantized tensor */\n'
            + '\n'.join(quant_lines)
            + '\n\n'
            + '\n'.join(helper_lines)
            + '\n'
        )
    else:
        quant_constants_str = (
            '/* Model uses float I/O — no quantization helpers needed. */\n'
        )
    
    # Kernel-specific includes
    extra_kernel_includes = ''
    if 9 in ops and not use_npu:
        extra_kernel_includes += '#include "tensorflow/lite/micro/kernels/fully_connected.h"\n'
    if 25 in ops and not use_npu:
        extra_kernel_includes += '#include "tensorflow/lite/micro/kernels/softmax.h"\n'

    # NPU-specific includes
    npu_includes = ''
    if use_npu:
        npu_includes = '#include "tensorflow/lite/micro/kernels/ethosu.h"\n'
    
    # Header file
    header = f"""\
// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* Initialize the TFLM interpreter. Call once before {fn_run}(). */
int {fn_init}(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void {fn_run}(bool clean_outputs);

/* Model input pointers - write your input data here before calling {fn_run}() */
{input_decls_str}

/* Model output pointers - read results here after {fn_run}() returns */
{output_decls_str}

/* Arena debug getters - available after {fn_init}() returns 0 */
size_t {fn_get_arena_size}(void);
size_t {fn_get_arena_used}(void);
int    {fn_get_last_status}(void);

{quant_constants_str}
#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    
    # Source file
    source = f"""\
// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). TFLM, CMSIS-NN, and Ethos-U
// runtime components are supplied separately via FSP under their own
// respective licenses.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "{h_file}"
#include "{data_header}"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
{extra_kernel_includes}
{npu_includes}
/*
 * Tensor arena size (estimated). If AllocateTensors() fails, increase this.
 * Operator scratch buffers are hardware-dependent (CMSIS-NN vs reference)
 * and cannot be predicted exactly from Python.
 */
#define {arena_macro} {arena_size}
__attribute__((aligned(16))) static uint8_t {arena_var}[{arena_macro}];

extern "C" {{

/* TFLM globals */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input_tensor[{len(inputs)}];
static TfLiteTensor* model_output_tensor[{len(outputs)}];
static size_t g_arena_used_bytes = 0;
static int    g_last_invoke_status = 0;

int {fn_init}(void) {{
  static tflite::MicroMutableOpResolver<{num_ops}> resolver;
{op_resolver_block}

  const tflite::Model* model = tflite::GetModel({var_data});
  if (model == nullptr) {{
    return -1;
  }}

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, {arena_var}, {arena_macro});
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {{
    return -2;
  }}
  g_arena_used_bytes = interpreter->arena_used_bytes();

  /* Cache input/output tensor pointers */
{chr(10).join(f'  model_input_tensor[{i}] = interpreter->input({i});' for i in range(len(inputs)))}
{chr(10).join(f'  model_output_tensor[{i}] = interpreter->output({i});' for i in range(len(outputs)))}

  return 0;
}}

void {fn_run}(bool clean_outputs) {{
  if (clean_outputs) {{
{chr(10).join(f'    memset(model_output_tensor[{i}]->data.raw, 0, model_output_tensor[{i}]->bytes);' for i in range(len(outputs)))}
  }}

  g_last_invoke_status = (interpreter->Invoke() == kTfLiteOk) ? 0 : -1;
}}

/* Arena debug getters */
size_t {fn_get_arena_size}(void) {{ return (size_t){arena_macro}; }}
size_t {fn_get_arena_used}(void) {{ return g_arena_used_bytes; }}
int    {fn_get_last_status}(void)     {{ return g_last_invoke_status; }}

/* Input pointer accessors */
{input_impls_str}

/* Output pointer accessors */
{output_impls_str}

}} /* extern "C" */
"""
    
    (output_dir / h_file).write_text(header, encoding='utf-8')
    (output_dir / cc_file).write_text(source, encoding='utf-8')


# =============================================================================
# TFLM Backend - Main Compilation Flow
# =============================================================================

def _validate_tflm_options(args) -> None:
    """Reject options that are unsupported or inapplicable to the TFLM backend."""
    invalid = []
    if getattr(args, 'host_evaluate', False) and getattr(args, 'npu', False):
        invalid.append(
            "--host-evaluate supports TFLM CPU models only; a desktop TFLite "
            "interpreter cannot execute the Ethos-U custom operator"
        )
    if getattr(args, 'cpu', False):
        if getattr(args, 'memory_mode', 'Sram_Only') != 'Sram_Only':
            invalid.append("--memory-mode is a Vela/NPU option and is not valid for TFLM CPU")
        if getattr(args, 'optimization', None) not in (None, 'Performance'):
            invalid.append("--optimization is a Vela/NPU option and is not valid for TFLM CPU")
    if not getattr(args, 'quantize', False):
        if getattr(args, 'calib_data', ''):
            invalid.append("--calib-data requires --quantize")
        if getattr(args, 'calib_num', 5) != 5:
            invalid.append("--calib-num requires --quantize")

    if invalid:
        details = '\n'.join(f"  - {message}" for message in invalid)
        raise ValueError(f"Invalid option(s) for the TFLM backend:\n{details}")

def compile_model_tflm(
    model_path: Path,
    output_dir: Path,
    args,
    needs_external: Optional[bool] = None,
) -> CompileResult:
    """
    Main compilation flow for TFLM backend.
    
    Generates C source files with unified RunModel(bool) API
    using TensorFlow Lite for Microcontrollers.

    needs_external: external-memory decision already made by the caller.
    main() passes it so the output folder name and the build always agree;
    when None, the decision is made here from the final model size.
    """
    result = CompileResult(
        model_name=model_path.stem,
        model_path=model_path,
        output_dir=output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Compiling (TFLM): {model_path.name}")
    print(f"{'='*60}")
    
    try:
        _validate_tflm_options(args)
        model_format = get_model_format(model_path)
        
        is_quantized = is_model_quantized(model_path)
        use_npu = args.npu
        use_quantize = args.quantize
        
        print(f"  Format: {model_format}")
        print(f"  Quantized: {is_quantized}")
        print(f"  Target: {'NPU (Ethos-U55)' if use_npu else 'CPU (TFLM)'}")

        # Keep a copy of the original input model for traceability and
        # post-build metrics tooling (e.g. MAC counting from FlatBuffer).
        ref_dir = output_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_model_path = ref_dir / model_path.name
        try:
            shutil.copy2(model_path, ref_model_path)
            print(f"  Reference model saved: {ref_model_path}")
        except Exception as e:
            print(f"  Warning: Could not save reference model copy: {e}")
        
        # Quantize if needed
        active_model_path = model_path
        
        if use_quantize and not is_quantized:
            print("  Quantizing FP32 model to INT8...")
            qtz_dir = output_dir / "quantization"
            calib_path = Path(args.calib_data) if args.calib_data else None
            active_model_path = quantize_tflite_model(
                model_path, qtz_dir, calib_path, args.calib_num
            )
            is_quantized = True
        
        if use_npu and not is_quantized:
            print_warning("  ⚠️  WARNING: Model is not fully quantized for Ethos-U acceleration.")
            print("           Vela will keep non-quantized operators on the CPU; NPU acceleration")
            print("           may be partial or zero. Use --quantize for Ethos-U execution.")
        
        # External memory detection
        file_size_bytes = active_model_path.stat().st_size
        if needs_external is None:
            needs_ext_mem = args.external or needs_external_memory(file_size_bytes, args.memory_threshold)
        else:
            needs_ext_mem = needs_external
        result.needs_external_memory = needs_ext_mem
        
        if needs_ext_mem:
            print(f"  External memory: enabled ({file_size_bytes / (1024*1024):.2f} MB)")
        
        # Run Vela for NPU target
        vela_sram_bytes = None
        if use_npu:
            print("  Compiling with Vela for Ethos-U55...")
            vela_dir = output_dir / "vela"
            vela_config = {
                'enable_ospi': needs_ext_mem,
                'sys_config': 'RA8P1',
                'memory_mode': args.memory_mode,
                'accel_config': 'ethos-u55-256',
                'optimise': args.optimization,
            }
            active_model_path, vela_sram_bytes = run_vela(active_model_path, vela_dir, vela_config)
            print(f"  Vela output: {active_model_path}")
        
        # Parse the final .tflite for metadata
        # Skip host estimation only when Vela supplied an NPU SRAM requirement.
        # An all-CPU Vela result has no SRAM summary and uses the normal CPU
        # heuristic; mixed/custom-op models fall back to the existing heuristic.
        print("  Parsing model structure...")
        model_info = parse_tflite_model(
            active_model_path,
            skip_arena_estimate=(use_npu and vela_sram_bytes is not None),
        )
        
        print(f"  Ops: {[TFLITE_BUILTIN_OP_MAP.get(o, (f'OP_{o}',))[0] for o in model_info['ops']]}")
        print(f"  Inputs: {[(i['name'], i['shape']) for i in model_info['inputs']]}")
        print(f"  Outputs: {[(o['name'], o['shape']) for o in model_info['outputs']]}")

        # Checked on the final graph: after Vela only CPU-fallback operators
        # remain, and none of the C sources has been written yet.
        _require_supported_builtin_ops(model_info['ops'])

        if vela_sram_bytes is not None:
            overhead = _tflm_npu_arena_overhead(model_info)
            vela_sram_bytes += overhead
            print(f"  NPU arena: {vela_sram_bytes} bytes (Vela SRAM + {overhead} B TFLM "
                  f"overhead for {model_info['num_tensors']} tensors, "
                  f"{len(model_info['inputs']) + len(model_info['outputs'])} I/O)")
        
        # Generate C source files
        deploy_dir = output_dir / "deploy" / "src"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        model_suffix = args.suffix if hasattr(args, 'suffix') else ''
        print("  Generating C source files...")
        _generate_model_data_files(active_model_path, deploy_dir, model_info, suffix=model_suffix)
        _generate_model_files(deploy_dir, model_info, use_npu, arena_size=vela_sram_bytes, suffix=model_suffix)
        
        print(f"  C-code generated in: {deploy_dir}")
        print("  ✅ Compilation complete")
        
        # Host evaluation via TFLite interpreter
        if args.host_evaluate:
            print("  Running host evaluation via TFLite interpreter...")
            success, err = _tflm_host_evaluate(active_model_path, model_info)
            if not success:
                result.status = Status.ERROR_VALIDATION
                result.last_error = f"Host evaluation failed: {err}"
                return result
        
        result.status = Status.SUCCESS
        print("  SUCCESS")
        
    except ValueError as e:
        result.status = Status.ERROR_DEPLOY
        result.last_error = str(e)
        print(f"  VALIDATION ERROR: {e}")
    except RuntimeError as e:
        result.status = Status.ERROR_DEPLOY
        result.last_error = str(e)
        print(f"  ERROR: {e}")
    except Exception as e:
        result.status = Status.ERROR_DEPLOY
        result.last_error = str(e)
        print(f"  ERROR: {e}")
    
    return result


def _tflm_host_evaluate(model_path: Path, model_info: Dict) -> Tuple[bool, str]:
    """
    Run a CPU .tflite model once as a host invocation smoke test.
    """
    try:
        import tensorflow as tf
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflri
            interp = tflri.Interpreter(model_path=str(model_path))
        except ImportError:
            message = "Neither tensorflow nor tflite_runtime is installed"
            print(f"    Error: {message}; host evaluation was requested")
            return (False, message)
    else:
        interp = tf.lite.Interpreter(model_path=str(model_path))

    try:
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        for inp in input_details:
            shape = tuple(int(dim) if int(dim) > 0 else 1 for dim in inp['shape'])
            dtype = np.dtype(inp['dtype'])
            if np.issubdtype(dtype, np.bool_):
                data = np.random.randint(0, 2, size=shape).astype(dtype)
            elif np.issubdtype(dtype, np.unsignedinteger):
                info = np.iinfo(dtype)
                high = min(int(info.max), 255) + 1
                data = np.random.randint(0, high, size=shape).astype(dtype)
            elif np.issubdtype(dtype, np.signedinteger):
                info = np.iinfo(dtype)
                low = max(int(info.min), -128)
                high = min(int(info.max), 127) + 1
                data = np.random.randint(low, high, size=shape).astype(dtype)
            elif np.issubdtype(dtype, np.floating):
                data = np.random.uniform(0.0, 1.0, size=shape).astype(dtype)
            else:
                raise TypeError(f"Unsupported host input dtype {dtype} for '{inp['name']}'")
            interp.set_tensor(inp['index'], data)

        interp.invoke()

        for out in output_details:
            output = interp.get_tensor(out['index'])
            print(f"    Output '{out['name']}': shape={output.shape}, dtype={output.dtype}")

        print("    Host invocation smoke test passed")
        return (True, "")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"    Host invocation smoke test failed: {message}")
        return (False, message)



# =============================================================================
# JUnit XML Output
# =============================================================================

def write_junit_xml(results: List[CompileResult], output_path: Path):
    """Write JUnit XML report."""
    try:
        from junitparser import TestCase, TestSuite, JUnitXml, Error
    except ImportError:
        print("Warning: junitparser not installed, skipping XML output")
        return
    
    suite = TestSuite(output_path.stem)
    
    for result in results:
        # Use the output folder name (e.g. model_NPU_external_quantized) as case name
        # so report consumers can match it directly to the deploy folder.
        case_name = result.output_dir.name + result.model_path.suffix
        test_case = TestCase(case_name)
        if result.status != Status.SUCCESS:
            test_case.result = [Error(result.last_error, result.status.name)]
        suite.add_testcase(test_case)
    
    xml = JUnitXml()
    xml.add_testsuite(suite)
    xml.write(str(output_path))
    print(f"JUnit XML written to: {output_path}")


# =============================================================================
# YAML Configuration Loading
# =============================================================================

YAML_CONFIG_KEYS = frozenset({
    'model_path', 'output_dir', 'target', 'quantize', 'external',
    'memory_threshold', 'calib_data', 'calib_num', 'memory_mode',
    'optimization', 'suffix', 'host_evaluate', 'result',
    # Optional; only 'tflm' is valid here.
    'compiler',
})

def _parse_yaml_bool(value, field_name: str) -> bool:
    """Parse YAML bool values with support for common string/int representations."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        print(f"Error: '{field_name}' must be a boolean (true/false), got integer: {value}")
        sys.exit(1)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', 'yes', 'y', 'on', '1'):
            return True
        if normalized in ('false', 'no', 'n', 'off', '0'):
            return False
        print(f"Error: '{field_name}' must be a boolean, got string: '{value}'")
        sys.exit(1)

    print(f"Error: '{field_name}' must be a boolean, got type: {type(value).__name__}")
    sys.exit(1)


def load_config(config_path: str) -> Namespace:
    """
    Load compile configuration from a YAML file and return an args-compatible Namespace.

    Expected YAML structure:
        model_path: /path/to/model.tflite
        output_dir: /path/to/output
        target: npu          # 'cpu' or 'npu'  (required)
        compiler: tflm       # optional; any other value is rejected
        quantize: false
        external: false
        memory_threshold: 0.8
        calib_data: ''
        calib_num: 5
        memory_mode: Sram_Only
        optimization: Performance
        suffix: ''
        host_evaluate: false
        result: ''
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Install it with: pip install pyyaml")
        sys.exit(1)

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file does not exist: {config_file}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        print("Error: Config file is empty.")
        sys.exit(1)

    # Keys this script never reads (e.g. typos) would otherwise be dropped
    # without notice.
    unknown_keys = sorted(str(k) for k in cfg if k not in YAML_CONFIG_KEYS)
    if unknown_keys:
        print_warning(f"Warning: ignoring unknown key(s) in {config_file.name}: "
                      f"{', '.join(unknown_keys)}")

    # -------------------------------------------------------------------------
    # Validate required fields
    # -------------------------------------------------------------------------
    if 'model_path' not in cfg:
        print("Error: 'model_path' is required in the config file.")
        sys.exit(1)
    if 'output_dir' not in cfg:
        print("Error: 'output_dir' is required in the config file.")
        sys.exit(1)

    target = str(cfg.get('target', '')).lower()
    if target not in ('cpu', 'npu'):
        print("Error: 'target' must be either 'cpu' or 'npu' in the config file.")
        sys.exit(1)

    compiler = str(cfg.get('compiler', 'tflm')).lower()
    if compiler != 'tflm':
        print(f"Error: 'compiler: {compiler}' is not supported by this script, which "
              "only compiles for TFLM. Use the matching backend script instead.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Validate choice fields
    # -------------------------------------------------------------------------
    memory_mode = cfg.get('memory_mode', 'Sram_Only')
    if memory_mode not in ('Sram_Only', 'Shared_Sram'):
        print(f"Error: 'memory_mode' must be 'Sram_Only' or 'Shared_Sram', got: '{memory_mode}'")
        sys.exit(1)

    optimization = cfg.get('optimization')
    if 'optimization' in cfg and optimization not in ('Performance', 'Size'):
        print(f"Error: 'optimization' must be 'Performance' or 'Size', got: '{optimization}'")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Build Namespace that mirrors the argparse output
    # -------------------------------------------------------------------------
    args = Namespace(
        model_path=str(cfg['model_path']),
        output_dir=str(cfg['output_dir']),
        # target flags
        cpu=(target == 'cpu'),
        npu=(target == 'npu'),
        # quantization
        quantize=_parse_yaml_bool(cfg.get('quantize', False), 'quantize'),
        # external memory
        external=_parse_yaml_bool(cfg.get('external', False), 'external'),
        memory_threshold=float(cfg.get('memory_threshold', DEFAULT_OSPI_THRESHOLD_MB)),
        # calibration
        calib_data=str(cfg.get('calib_data', '')),
        calib_num=int(cfg.get('calib_num', 5)),
        # Vela configuration
        memory_mode=memory_mode,
        optimization=optimization,
        suffix=str(cfg.get('suffix', '')),
        # evaluation
        host_evaluate=_parse_yaml_bool(cfg.get('host_evaluate', False), 'host_evaluate'),
        # output
        result=str(cfg.get('result', '')),
    )

    return args


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_parser() -> ArgumentParser:
    """Create argument parser."""
    parser = ArgumentParser(
        prog='mcu_compile_tflm.py',
        description='MCU model compilation script for the TFLM backend (TensorFlow Lite Micro + Vela/CMSIS-NN)'
    )

    # Positional arguments (optional if YAML config is used)
    parser.add_argument('model_path', type=str, nargs='?',
                        help='Path to .tflite model file or directory, or path to YAML config file')
    parser.add_argument('output_dir', type=str, nargs='?',
                        help='Output directory for compiled C-code')

    # Target platform (required for CLI mode)
    target_group = parser.add_mutually_exclusive_group(required=False)
    target_group.add_argument('--cpu', action='store_true',
                              help='Deploy for CPU (CMSIS-NN)')
    target_group.add_argument('--npu', action='store_true',
                              help='Deploy for NPU (Ethos-U55, requires INT8)')

    # Quantization (opt-in: without --quantize, model deploys as-is)
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize FP32 model to INT8 before deploy')

    # External memory flag
    parser.add_argument('--external', action='store_true',
                        help='Force external memory mode (enables Vela OSPI for NPU)')

    # Memory threshold for auto-detection
    parser.add_argument('--memory-threshold', type=float, default=DEFAULT_OSPI_THRESHOLD_MB,
                        help='Threshold in MB for auto-detecting external memory need '
                             f'(default: {DEFAULT_OSPI_THRESHOLD_MB})')

    # Quantization options
    parser.add_argument('--calib-data', type=str, default='',
                        help='Path to calibration data (.npy file or directory)')
    parser.add_argument('--calib-num', type=int, default=5,
                        help='Number of calibration samples (default: 5)')

    # Vela configuration
    parser.add_argument('--memory-mode', type=str, default='Sram_Only',
                        choices=['Sram_Only', 'Shared_Sram'],
                        help='Vela memory mode (default: Sram_Only)')
    # Keep omission distinct so the NPU default does not invalidate CPU commands.
    parser.add_argument('--optimization', type=str, default=None,
                        choices=['Performance', 'Size'],
                        help='Vela optimization target (NPU only; default: Performance)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for generated symbols, file names and the output folder')

    # Evaluation options
    parser.add_argument('--host-evaluate', action='store_true',
                        help='Host evaluation: TFLM CPU invocation smoke test')

    # Output options
    parser.add_argument('--result', type=str, default='',
                        help='Path for JUnit XML output')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = create_parser()

    # Check if first argument is a YAML config file
    if len(sys.argv) >= 2 and Path(sys.argv[1]).suffix.lower() in ('.yaml', '.yml'):
        print("Using YAML configuration mode...")
        args = load_config(sys.argv[1])
        input_mode = 'yaml'
    else:
        # Parse CLI arguments
        args = parser.parse_args()
        input_mode = 'cli'

        # Validate CLI mode: require both model_path and output_dir, and target
        if not args.model_path or not args.output_dir:
            print("Error: model_path and output_dir are required in CLI mode")
            parser.print_help()
            sys.exit(1)

        if not args.cpu and not args.npu:
            print("Error: --cpu or --npu is required in CLI mode")
            parser.print_help()
            sys.exit(1)

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    if model_path.is_file():
        model_files = [model_path]
    else:
        model_files = list(model_path.rglob("*.tflite"))

    if len(model_files) == 0:
        print(f"Error: No models found in: {model_path}")
        sys.exit(1)

    print(f"Found {len(model_files)} model(s)")
    print(f"Input mode: {input_mode.upper()}")

    results = []
    for model_file in model_files:
        target_str = "NPU" if args.npu else "CPU"

        # Decide external memory once, here, and pass it to the compile so the
        # folder name and the build can never disagree. Only a model that will
        # actually be quantized (FP32 + --quantize) is estimated at 1/4 size.
        will_quantize = False
        if args.quantize:
            try:
                will_quantize = not is_model_quantized(model_file)
            except ValueError:
                pass  # unsupported format; compile_model_tflm reports it
        model_size = model_file.stat().st_size
        effective_size = model_size // 4 if will_quantize else model_size
        needs_ext = args.external or needs_external_memory(effective_size, args.memory_threshold)

        ext_str = "_external" if needs_ext else ""
        quant_str = "_quantized" if will_quantize else ""

        safe_stem = model_file.stem.replace(' ', '_')
        sfx = _normalize_suffix(args.suffix)
        sfx_str = f"_{sfx}" if sfx else ""
        base_dir_name = f"{safe_stem}_{target_str}_tflm{ext_str}{quant_str}{sfx_str}"
        model_output_dir = output_dir / base_dir_name

        result = compile_model_tflm(model_file, model_output_dir, args,
                                    needs_external=needs_ext)
        results.append(result)

        # Force garbage collection to prevent memory buildup
        gc.collect()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r.status == Status.SUCCESS)
    print(f"Success: {success_count}/{len(results)}")

    for result in results:
        status_str = "SUCCESS" if result.status == Status.SUCCESS else f"ERROR: {result.last_error}"
        ext_str = " [EXTERNAL]" if result.needs_external_memory else ""
        print(f"  {result.model_name}: {status_str}{ext_str}")

    if args.result:
        write_junit_xml(results, Path(args.result))

    if success_count < len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
