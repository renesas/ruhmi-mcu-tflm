// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef MODEL_PALM_H
#define MODEL_PALM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
struct TfLiteTensor;
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel_palm(). */
int ModelInit_palm(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel_palm(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel_palm() */
int8_t* GetModelInputPtr_palm_serving_default_input_0();

/* Model output pointers - read results here after RunModel_palm() returns */
int8_t* GetModelOutputPtr_palm_PartitionedCall_0();
int8_t* GetModelOutputPtr_palm_PartitionedCall_3();
int8_t* GetModelOutputPtr_palm_PartitionedCall_2();
int8_t* GetModelOutputPtr_palm_PartitionedCall_1();

/* Arena debug getters - available after ModelInit_palm() returns 0 */
size_t GetTensorArenaSizeBytes_palm(void);
size_t GetTensorArenaUsedBytes_palm(void);
int    GetLastInvokeStatus_palm(void);
const struct TfLiteTensor* GetModelInputTensor_palm(void);
const struct TfLiteTensor* GetModelOutputTensor_palm(size_t index);

/* Quantization parameters (scale / zero_point) for each quantized tensor */
#define PALM_INPUT_SCALE_SERVING_DEFAULT_INPUT_0              0.003921568394f
#define PALM_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0          (-128)
#define PALM_OUTPUT_SCALE_PARTITIONEDCALL_0                   0.007054821122f
#define PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0               (-80)
#define PALM_OUTPUT_SCALE_PARTITIONEDCALL_3                   0.00390625f
#define PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_3               (-128)
#define PALM_OUTPUT_SCALE_PARTITIONEDCALL_2                   0.005439953879f
#define PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2               (-106)
#define PALM_OUTPUT_SCALE_PARTITIONEDCALL_1                   0.005439953879f
#define PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1               (-106)

/* Typed per-tensor quantization helpers. */
static inline int8_t model_quantize_palm_input_serving_default_input_0(float x) {
    int64_t v = (int64_t)roundf(x / PALM_INPUT_SCALE_SERVING_DEFAULT_INPUT_0) + PALM_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}
static inline float model_dequantize_palm_output_PartitionedCall_0(int8_t x) {
    return ((float)(x) - (float)PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0) * PALM_OUTPUT_SCALE_PARTITIONEDCALL_0;
}
static inline float model_dequantize_palm_output_PartitionedCall_3(int8_t x) {
    return ((float)(x) - (float)PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_3) * PALM_OUTPUT_SCALE_PARTITIONEDCALL_3;
}
static inline float model_dequantize_palm_output_PartitionedCall_2(int8_t x) {
    return ((float)(x) - (float)PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2) * PALM_OUTPUT_SCALE_PARTITIONEDCALL_2;
}
static inline float model_dequantize_palm_output_PartitionedCall_1(int8_t x) {
    return ((float)(x) - (float)PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1) * PALM_OUTPUT_SCALE_PARTITIONEDCALL_1;
}
/* Backward-compatible helper for homogeneous quantized inputs. */
static inline int8_t model_quantize_palm(float x, float scale, int zero_point) {
    int64_t v = (int64_t)roundf(x / scale) + (int64_t)zero_point;
    if (v < -128) v = -128;
    if (v >  127) v = 127;
    return (int8_t)v;
}
/* Backward-compatible helper for homogeneous quantized outputs. */
static inline float model_dequantize_palm(int8_t x, float scale, int zero_point) {
    return ((float)(x) - (float)zero_point) * scale;
}

#ifdef __cplusplus
}
#endif

#endif /* MODEL_PALM_H */
