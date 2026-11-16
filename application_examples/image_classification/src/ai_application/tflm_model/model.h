// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel(). */
int ModelInit(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel() */
int8_t* GetModelInputPtr_serving_default_input_2_0();

/* Model output pointers - read results here after RunModel() returns */
int8_t* GetModelOutputPtr_StatefulPartitionedCall_0();

/* Arena debug getters - available after ModelInit() returns 0 */
size_t GetTensorArenaSizeBytes(void);
size_t GetTensorArenaUsedBytes(void);
int    GetLastInvokeStatus(void);
TfLiteTensor* GetModelInputTensor(void);
TfLiteTensor* GetModelOutputTensor(void);

/* Quantization parameters (scale / zero_point) for each quantized tensor */
#define INPUT_SCALE_SERVING_DEFAULT_INPUT_2_0            0.007843137719f
#define INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_2_0        (-1)
#define OUTPUT_SCALE_STATEFULPARTITIONEDCALL_0           0.00390625f
#define OUTPUT_ZERO_POINT_STATEFULPARTITIONEDCALL_0       (-128)

/* Typed per-tensor quantization helpers. */
static inline int8_t model_quantize_input_serving_default_input_2_0(float x) {
    int64_t v = (int64_t)roundf(x / INPUT_SCALE_SERVING_DEFAULT_INPUT_2_0) + INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_2_0;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}
static inline float model_dequantize_output_StatefulPartitionedCall_0(int8_t x) {
    return ((float)(x) - (float)OUTPUT_ZERO_POINT_STATEFULPARTITIONEDCALL_0) * OUTPUT_SCALE_STATEFULPARTITIONEDCALL_0;
}
/* Backward-compatible helper for homogeneous quantized inputs. */
static inline int8_t model_quantize(float x, float scale, int zero_point) {
    int64_t v = (int64_t)roundf(x / scale) + (int64_t)zero_point;
    if (v < -128) v = -128;
    if (v >  127) v = 127;
    return (int8_t)v;
}
/* Backward-compatible helper for homogeneous quantized outputs. */
static inline float model_dequantize(int8_t x, float scale, int zero_point) {
    return ((float)(x) - (float)zero_point) * scale;
}

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
