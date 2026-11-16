// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel(). */
int ModelInit(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel() */
int8_t* GetModelInputPtr_image_input();

/* Model output pointers - read results here after RunModel() returns */
int8_t* GetModelOutputPtr_Identity();
int8_t* GetModelOutputPtr_Identity_1();

/* Arena debug getters - available after ModelInit() returns 0 */
size_t GetTensorArenaSizeBytes(void);
size_t GetTensorArenaUsedBytes(void);
int    GetLastInvokeStatus(void);

/* Quantization parameters (scale / zero_point) for each quantized tensor */
#define INPUT_SCALE_IMAGE_INPUT                          0.003921176773f
#define INPUT_ZERO_POINT_IMAGE_INPUT                      (-128)
#define OUTPUT_SCALE_IDENTITY                            0.1340839118f
#define OUTPUT_ZERO_POINT_IDENTITY                        (47)
#define OUTPUT_SCALE_IDENTITY_1                          0.1853592545f
#define OUTPUT_ZERO_POINT_IDENTITY_1                      (10)

/* Typed per-tensor quantization helpers. */
static inline int8_t model_quantize_input_image_input(float x) {
    int64_t v = (int64_t)roundf(x / INPUT_SCALE_IMAGE_INPUT) + INPUT_ZERO_POINT_IMAGE_INPUT;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}
static inline float model_dequantize_output_Identity(int8_t x) {
    return ((float)(x) - (float)OUTPUT_ZERO_POINT_IDENTITY) * OUTPUT_SCALE_IDENTITY;
}
static inline float model_dequantize_output_Identity_1(int8_t x) {
    return ((float)(x) - (float)OUTPUT_ZERO_POINT_IDENTITY_1) * OUTPUT_SCALE_IDENTITY_1;
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
