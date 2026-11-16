// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). TFLM, CMSIS-NN, and Ethos-U
// runtime components are supplied separately via FSP under their own
// respective licenses.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "model_landmark.h"
#include "model_data_landmark.h"
#include "../tflm_memory.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/kernels/ethosu.h"

TFLM_ARENA_ATTRIBUTES static uint8_t persistent_arena[TFLM_PERSISTENT_BYTES];

extern "C" {

/* TFLM globals */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input_tensor[1];
static TfLiteTensor* model_output_tensor[3];
static size_t g_arena_used_bytes = 0;
static int    g_last_invoke_status = -1;

int ModelInit_landmark(void) {
  if (interpreter != nullptr) return 0;
  static bool attempted = false;
  if (attempted) return -1;
  attempted = true;
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddCustom(tflite::GetString_ETHOSU(), tflite::Register_ETHOSU()) != kTfLiteOk) {
    return -1;
  }

  const tflite::Model* model = tflite::GetModel(g_model_data_landmark);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    return -1;
  }

  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      persistent_arena, sizeof(persistent_arena),
      g_tflm_shared_scratch, TFLM_SHARED_SCRATCH_BYTES);
  if (allocator == nullptr) return -2;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, allocator);

  if (static_interpreter.AllocateTensors() != kTfLiteOk) {
    return -2;
  }
  interpreter = &static_interpreter;
  g_arena_used_bytes = interpreter->arena_used_bytes();

  /* Cache input/output tensor pointers */
  model_input_tensor[0] = interpreter->input(0);
  model_output_tensor[0] = interpreter->output(0);
  model_output_tensor[1] = interpreter->output(1);
  model_output_tensor[2] = interpreter->output(2);

  return 0;
}

void RunModel_landmark(bool clean_outputs) {
  if (interpreter == nullptr) {
    g_last_invoke_status = -1;
    return;
  }
  if (clean_outputs) {
    memset(model_output_tensor[0]->data.raw, 0, model_output_tensor[0]->bytes);
    memset(model_output_tensor[1]->data.raw, 0, model_output_tensor[1]->bytes);
    memset(model_output_tensor[2]->data.raw, 0, model_output_tensor[2]->bytes);
  }

  g_last_invoke_status = (interpreter->Invoke() == kTfLiteOk) ? 0 : -1;
}

/* Arena debug getters */
size_t GetTensorArenaSizeBytes_landmark(void) { return TFLM_SHARED_SCRATCH_BYTES + sizeof(persistent_arena); }
size_t GetTensorArenaUsedBytes_landmark(void) { return g_arena_used_bytes; }
int    GetLastInvokeStatus_landmark(void)     { return g_last_invoke_status; }
const TfLiteTensor* GetModelInputTensor_landmark(void) { return model_input_tensor[0]; }
const TfLiteTensor* GetModelOutputTensor_landmark(size_t index) {
  return index < 3 ? model_output_tensor[index] : nullptr;
}

/* Input pointer accessors */
int8_t* GetModelInputPtr_landmark_serving_default_input_0() {
  return interpreter ? model_input_tensor[0]->data.int8 : nullptr;
}

/* Output pointer accessors */
int8_t* GetModelOutputPtr_landmark_PartitionedCall_1() {
  return interpreter ? model_output_tensor[0]->data.int8 : nullptr;
}

int8_t* GetModelOutputPtr_landmark_PartitionedCall_0() {
  return interpreter ? model_output_tensor[1]->data.int8 : nullptr;
}

int8_t* GetModelOutputPtr_landmark_PartitionedCall_2() {
  return interpreter ? model_output_tensor[2]->data.int8 : nullptr;
}

} /* extern "C" */
