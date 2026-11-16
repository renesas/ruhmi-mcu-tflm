// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). TFLM, CMSIS-NN, and Ethos-U
// runtime components are supplied separately via FSP under their own
// respective licenses.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "model.h"
#include "model_data.h"
#include "application_config.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/kernels/ethosu.h"

/*
 * Tensor arena size (estimated). If AllocateTensors() fails, increase this.
 * Operator scratch buffers are hardware-dependent (CMSIS-NN vs reference)
 * and cannot be predicted exactly from Python.
 */
#define TENSOR_ARENA_SIZE 422688
#if (TENSOR_ARENA_ALLOCATION == ALLOCATE_TO_ONCHIP_RAM)
__attribute__((aligned(16))) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
#elif (TENSOR_ARENA_ALLOCATION == ALLOCATE_TO_SDRAM)
__attribute__((aligned(16), section(".sdram_noinit"))) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
#else
#error "Unsupported TENSOR_ARENA_ALLOCATION"
#endif

extern "C" {

/* TFLM globals */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input_tensor[1];
static TfLiteTensor* model_output_tensor[1];
static size_t g_arena_used_bytes = 0;
static int    g_last_invoke_status = -1;

int ModelInit(void) {
  if (interpreter != nullptr) {
    return 0;
  }
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddCustom(tflite::GetString_ETHOSU(), tflite::Register_ETHOSU()) != kTfLiteOk) {
    return -1;
  }

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model == nullptr) {
    return -1;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
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

void RunModel(bool clean_outputs) {
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
size_t GetTensorArenaSizeBytes(void) { return (size_t)TENSOR_ARENA_SIZE; }
size_t GetTensorArenaUsedBytes(void) { return g_arena_used_bytes; }
int    GetLastInvokeStatus(void)     { return g_last_invoke_status; }
TfLiteTensor* GetModelInputTensor(void) { return model_input_tensor[0]; }
TfLiteTensor* GetModelOutputTensor(void) { return model_output_tensor[0]; }

/* Input pointer accessors */
int8_t* GetModelInputPtr_serving_default_input_2_0() {
  return model_input_tensor[0] ? model_input_tensor[0]->data.int8 : nullptr;
}

/* Output pointer accessors */
int8_t* GetModelOutputPtr_StatefulPartitionedCall_0() {
  return model_output_tensor[0] ? model_output_tensor[0]->data.int8 : nullptr;
}

} /* extern "C" */
