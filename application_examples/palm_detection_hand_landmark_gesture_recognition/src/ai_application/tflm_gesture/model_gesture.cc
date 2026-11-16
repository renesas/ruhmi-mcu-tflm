// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). TFLM, CMSIS-NN, and Ethos-U
// runtime components are supplied separately via FSP under their own
// respective licenses.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "model_gesture.h"
#include "model_data_gesture.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"


/* Gesture classification runs in the display thread, independently of the NPU. */
#define TENSOR_ARENA_SIZE_GESTURE 4096
__attribute__((aligned(32))) static uint8_t tensor_arena_gesture[TENSOR_ARENA_SIZE_GESTURE];

extern "C" {

/* TFLM globals */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input_tensor[1];
static TfLiteTensor* model_output_tensor[1];
static size_t g_arena_used_bytes = 0;
static int    g_last_invoke_status = -1;

int ModelInit_gesture(void) {
  if (interpreter != nullptr) return 0;
  static bool attempted = false;
  if (attempted) return -1;
  attempted = true;
  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    return -1;
  }

  const tflite::Model* model = tflite::GetModel(g_model_data_gesture);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    return -1;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena_gesture, TENSOR_ARENA_SIZE_GESTURE);
  if (static_interpreter.AllocateTensors() != kTfLiteOk) {
    return -2;
  }
  interpreter = &static_interpreter;
  g_arena_used_bytes = interpreter->arena_used_bytes();

  /* Cache input/output tensor pointers */
  model_input_tensor[0] = interpreter->input(0);
  model_output_tensor[0] = interpreter->output(0);

  return 0;
}

void RunModel_gesture(bool clean_outputs) {
  if (interpreter == nullptr) {
    g_last_invoke_status = -1;
    return;
  }
  if (clean_outputs) {
    memset(model_output_tensor[0]->data.raw, 0, model_output_tensor[0]->bytes);
  }

  g_last_invoke_status = (interpreter->Invoke() == kTfLiteOk) ? 0 : -1;
}

/* Arena debug getters */
size_t GetTensorArenaSizeBytes_gesture(void) { return (size_t)TENSOR_ARENA_SIZE_GESTURE; }
size_t GetTensorArenaUsedBytes_gesture(void) { return g_arena_used_bytes; }
int    GetLastInvokeStatus_gesture(void)     { return g_last_invoke_status; }
const TfLiteTensor* GetModelInputTensor_gesture(void) { return model_input_tensor[0]; }
const TfLiteTensor* GetModelOutputTensor_gesture(size_t index) {
  return index < 1 ? model_output_tensor[index] : nullptr;
}

/* Input pointer accessors */
float* GetModelInputPtr_gesture_input_1() {
  return interpreter ? model_input_tensor[0]->data.f : nullptr;
}

/* Output pointer accessors */
float* GetModelOutputPtr_gesture_Identity() {
  return interpreter ? model_output_tensor[0]->data.f : nullptr;
}

} /* extern "C" */
