# Known Issues and Workarounds

This directory describes known runtime problems, affected environments, workarounds, and applicable limitations.

| Document | Affected execution path | Description |
|----------|-------------------------|-------------|
| [Higher-rank INT8 Transpose](cmsis_nn_transpose_rank_limit.md) | CPU execution through the affected TFLM CMSIS-NN Transpose adapter, including CPU fallback in NPU applications. | Rank-5 and higher INT8 inputs require compatibility checks. The affected adapter can skip a five-dimensional Transpose while reporting success. Includes a complete reference-callback example for the five-dimensional case. |
| [Ethos-U cache maintenance overhead](ethosu_cache_maintenance_overhead.md) | FSP 6.6.0 CPU/NPU applications with D-cache enabled. | Redundant maintenance of read-only or overlapping base regions increases inference time. Includes measured impact, a manual filtering workaround, validation scope, and memory-layout limitations. |

For other problems, see [Troubleshooting](../tflm_backend/README.md#13-troubleshooting) and [Known limitations](../tflm_backend/README.md#14-known-limitations).
