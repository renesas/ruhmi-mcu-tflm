// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef MODEL_LANDMARK_H
#define MODEL_LANDMARK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
struct TfLiteTensor;
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel_landmark(). */
int ModelInit_landmark(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel_landmark(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel_landmark() */
int8_t* GetModelInputPtr_landmark_serving_default_input_0();

/* Model output pointers - read results here after RunModel_landmark() returns */
int8_t* GetModelOutputPtr_landmark_PartitionedCall_1();
int8_t* GetModelOutputPtr_landmark_PartitionedCall_0();
int8_t* GetModelOutputPtr_landmark_PartitionedCall_2();

/* Arena debug getters - available after ModelInit_landmark() returns 0 */
size_t GetTensorArenaSizeBytes_landmark(void);
size_t GetTensorArenaUsedBytes_landmark(void);
int    GetLastInvokeStatus_landmark(void);
const struct TfLiteTensor* GetModelInputTensor_landmark(void);
const struct TfLiteTensor* GetModelOutputTensor_landmark(size_t index);

/* Quantization parameters (scale / zero_point) for each quantized tensor */
#define LANDMARK_INPUT_SCALE_SERVING_DEFAULT_INPUT_0              0.003921568859f
#define LANDMARK_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0          (-128)
#define LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_1                   0.00390625f
#define LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1               (-128)
#define LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_0                   1.10766685f
#define LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0               (-43)
#define LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_2                   0.00390625f
#define LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2               (-128)

/* Typed per-tensor quantization helpers. */
static inline int8_t model_quantize_landmark_input_serving_default_input_0(float x) {
    int64_t v = (int64_t)roundf(x / LANDMARK_INPUT_SCALE_SERVING_DEFAULT_INPUT_0) + LANDMARK_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}
static inline float model_dequantize_landmark_output_PartitionedCall_1(int8_t x) {
    return ((float)(x) - (float)LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1) * LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_1;
}
static inline float model_dequantize_landmark_output_PartitionedCall_0(int8_t x) {
    return ((float)(x) - (float)LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0) * LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_0;
}
static inline float model_dequantize_landmark_output_PartitionedCall_2(int8_t x) {
    return ((float)(x) - (float)LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2) * LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_2;
}
/* Backward-compatible helper for homogeneous quantized inputs. */
static inline int8_t model_quantize_landmark(float x, float scale, int zero_point) {
    int64_t v = (int64_t)roundf(x / scale) + (int64_t)zero_point;
    if (v < -128) v = -128;
    if (v >  127) v = 127;
    return (int8_t)v;
}
/* Backward-compatible helper for homogeneous quantized outputs. */
static inline float model_dequantize_landmark(int8_t x, float scale, int zero_point) {
    return ((float)(x) - (float)zero_point) * scale;
}

#ifdef __cplusplus
}
#endif

#endif /* MODEL_LANDMARK_H */
