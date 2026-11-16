# FSP 6.6: Incorrect Inference for Models with Five-Dimensional INT8 Transpose

## Summary and Status

On FSP 6.6, generated CPU and CPU/NPU applications can silently produce incorrect results for models containing CPU-executed **INT8 `TRANSPOSE` operators with five-dimensional inputs**. Allocation and inference report success, but the affected Transpose operators do not write their output. The same NanoDet application worked with FSP 6.4.

The cause is the TFLM CMSIS-NN Transpose kernel adapter, which supports at most four dimensions for INT8 and hides the resulting error (see [Known Problem](#known-problem)). NPU applications are also affected when these Transpose operators fall back to the CPU.

> [!NOTE]
> **Status:** a reference-kernel workaround is described below; automatic compiler integration and independent target-side validation remain pending.

## Relevant Environment

| Item | Configuration |
|------|---------------|
| Example model | NanoDet INT8 object-detection model |
| Target | EK-RA8P1 |
| FSP integration | 6.6.0 |
| Vela | 5.2.0 |
| Previous working environment | FSP 6.4 |
| Affected application variants | CPU and CPU/NPU |

This configuration is an example deployment environment, not a restriction to NanoDet or EK-RA8P1. FSP 6.4 and 6.6 identify the software bundles used before and after the regression. The affected code is the TFLM CMSIS-NN adapter shipped in the FSP 6.6 bundle, not the FSP itself. Check the kernel implementation actually vendored in your project.

## Known Problem

The TFLM CMSIS-NN Transpose adapter (`tensorflow/lite/micro/kernels/cmsis_nn/transpose.cc`) has inconsistent rank checks and error propagation:

| Step | Behavior |
|------|----------|
| `TransposePrepare()` | Accepts input ranks up to five. |
| `TransposeEvalInt8()` | Supports at most four dimensions. For higher ranks it returns an error without writing the output. |
| Generic `TransposeEval()` | Ignores the `TransposeEvalInt8()` return value in its INT8 case and returns `kTfLiteOk`. |

Allocation and inference can therefore appear successful even though a Transpose operation was skipped. The CPU/NPU application is also affected because these Transpose operators still execute on the CPU.

Only the **INT8** path is affected. In the same adapter, float32 and int16 Transpose use the reference implementation, which supports five dimensions.

> This behavior was confirmed against upstream `tflite-micro` main. Confirm it against the exact sources vendored in FSP 6.6 before citing it.

For ranks above five, the retained `TransposePrepare()` rejects the input. This is a separate unsupported-rank condition, not the same silent five-dimensional failure.

## Scope and Symptoms

Rank means the number of dimensions of the tensor passed to `TRANSPOSE`, not the number of model inputs. For example, `[1, 2, 3, 4, 5]` has rank five.

The limitation is operator-specific and can affect any model using this INT8 execution path, not only NanoDet. It does not mean that every CMSIS-NN version, every data type (float32 and int16 Transpose are unaffected), or every higher-rank model fails in the same way. The affected operator is `TRANSPOSE`, not `TRANSPOSE_CONV` or `CONV_3D_TRANSPOSE`.

Selecting `--npu` does not prove that the operation runs on the NPU. Likewise, successful allocation, compilation, or invocation does not establish correct output. The compiler's `--host-evaluate` option executes desktop TFLite, not the embedded CMSIS-NN adapter, so it cannot rule out this defect.

## Proposed Workaround

For the five-dimensional INT8 case, keep the existing preparation function and replace the defective Invoke callback with a reference implementation. This bypasses the faulty INT8 execution path without editing dependency sources, changing model data, or changing Ethos-U registration.

> **Limitation:** the callback is INT8-only. Apply it only if every CPU-executed Transpose in the model has INT8 input and output. Otherwise it returns an error for those nodes. See [Check Applicability and Validate](#3-check-applicability-and-validate) for how to check.

<details>
<summary>Manual workaround: callback and registration</summary>

### 1. Add the Complete Callback

Add the following headers and callback to the generated `model.cc`, before `ModelInit()` and outside the existing `extern "C"` block:

```cpp
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/transpose.h"

static TfLiteStatus ReferenceTransposeEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor* permutation = tflite::micro::GetEvalInput(context, node, 1);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);

    const int32_t* permutation_data = tflite::micro::GetTensorData<int32_t>(permutation);
    tflite::TransposeParams params = {};
    params.perm_count = static_cast<int8_t>(permutation->dims->data[0]);
    for (int i = 0; i < params.perm_count; ++i) {
        params.perm[i] = permutation_data[i];
    }

    tflite::reference_ops::Transpose(
        params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
}
```

### 2. Replace the Transpose Registration

In `ModelInit()`, replace `resolver.AddTranspose();` with:

```cpp
resolver.AddTranspose(tflite::micro::RegisterOp(
    nullptr, tflite::TransposePrepare, ReferenceTransposeEval));
```

Leave the other operator registrations, including Ethos-U, unchanged. Check the registration's return status using the project's existing initialization error handling.

</details>

### 3. Check Applicability and Validate

**Type restriction:** this registration applies to every Transpose node handled by the resolver, so all of those inputs and outputs must be INT8. The callback returns an error for other types, so a mixed-type model fails loudly instead of producing silently wrong output. It is not a generic mixed-type implementation. Supporting such models requires a callback that dispatches on `input->type`, which this workaround intentionally omits.

**How to check:** open the final packaged model in a model viewer such as Netron. For NPU builds, open the post-Vela model. For each CPU-executed `TRANSPOSE`, confirm that the input tensor type is INT8 and note its rank. A model described only as "INT8" can still contain Transpose operators with other types.

**Rank restriction:** the example retains `TransposePrepare()` and supports ranks one through five. It does not add support for rank six or higher. Higher ranks require a separately validated preparation function and kernel implementation.

Build the modified project and verify its outputs on the target before relying on the workaround.

## Generated Code

The current [TFLM compiler](../../scripts/mcu_compile_tflm.py) uses the default Transpose registration and does not automatically apply this reference-kernel workaround. If you regenerate an affected project, check the generated registration and reapply any required changes.

The runtime components are supplied through the target FSP project. Updating host Python packages alone does not select or fix the embedded Transpose kernel.

## Target-Side Validation

To validate the workaround, compare target outputs with an independent reference using the same input and permutation. Use distinguishable tensor values and include rank-four control cases as well as rank-five cases. Record the dependency versions and actual kernel selection alongside the results.
