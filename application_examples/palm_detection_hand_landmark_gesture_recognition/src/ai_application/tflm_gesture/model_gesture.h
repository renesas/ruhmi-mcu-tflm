// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause).
#ifndef MODEL_GESTURE_H
#define MODEL_GESTURE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
struct TfLiteTensor;
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel_gesture(). */
int ModelInit_gesture(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel_gesture(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel_gesture() */
float* GetModelInputPtr_gesture_input_1();

/* Model output pointers - read results here after RunModel_gesture() returns */
float* GetModelOutputPtr_gesture_Identity();

/* Arena debug getters - available after ModelInit_gesture() returns 0 */
size_t GetTensorArenaSizeBytes_gesture(void);
size_t GetTensorArenaUsedBytes_gesture(void);
int    GetLastInvokeStatus_gesture(void);
const struct TfLiteTensor* GetModelInputTensor_gesture(void);
const struct TfLiteTensor* GetModelOutputTensor_gesture(size_t index);

/* Model uses float I/O — no quantization helpers needed. */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_GESTURE_H */
