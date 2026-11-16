#include "tflm_palm/model_palm.h"
#include "tflm_landmark/model_landmark.h"
#include "tflm_gesture/model_gesture.h"
#include "common/log_macros.h"
#include "tensorflow/lite/c/common.h"

template <size_t Rank>
static bool tensor_matches(const TfLiteTensor* tensor, TfLiteType type,
                           const int (&shape)[Rank], float scale = 0.0f,
                           int zero_point = 0)
{
    if (tensor == nullptr || tensor->data.raw == nullptr || tensor->type != type ||
        tensor->dims == nullptr || tensor->dims->size != Rank) {
        return false;
    }
    size_t elements = 1;
    for (size_t i = 0; i < Rank; ++i) {
        if (tensor->dims->data[i] != shape[i]) return false;
        elements *= static_cast<size_t>(shape[i]);
    }
    if (tensor->bytes != elements * (type == kTfLiteFloat32 ? sizeof(float) : sizeof(int8_t))) {
        return false;
    }
    return type == kTfLiteFloat32 ||
           (fabsf(tensor->params.scale - scale) <= scale * 1e-6f &&
            tensor->params.zero_point == zero_point);
}

extern "C" int palm_detection_init(void)
{
    static bool initialized = false;
    if (initialized) return 0;
    const int palm_status = ModelInit_palm();
    const int landmark_status = ModelInit_landmark();
    const int gesture_status = ModelInit_gesture();
    if (palm_status != 0 || landmark_status != 0 || gesture_status != 0) {
        error("TFLM initialization failed: palm=%d landmark=%d gesture=%d",
              palm_status, landmark_status, gesture_status);
        return -1;
    }

    const int palm_input[] = {1, 192, 192, 3};
    const int boxes[] = {2016, 4};
    const int scores[] = {2016};
    const int keypoints[] = {2016, 2};
    const int landmark_input[] = {1, 224, 224, 3};
    const int scalar[] = {1, 1};
    const int xyz[] = {1, 63};
    const int gesture_input[] = {1, 42};
    const int gesture_output[] = {1, 3};
    const bool valid =
        tensor_matches(GetModelInputTensor_palm(), kTfLiteInt8, palm_input,
                       PALM_INPUT_SCALE_SERVING_DEFAULT_INPUT_0, PALM_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0) &&
        tensor_matches(GetModelOutputTensor_palm(0), kTfLiteInt8, boxes,
                       PALM_OUTPUT_SCALE_PARTITIONEDCALL_0, PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0) &&
        tensor_matches(GetModelOutputTensor_palm(1), kTfLiteInt8, scores,
                       PALM_OUTPUT_SCALE_PARTITIONEDCALL_3, PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_3) &&
        tensor_matches(GetModelOutputTensor_palm(2), kTfLiteInt8, keypoints,
                       PALM_OUTPUT_SCALE_PARTITIONEDCALL_2, PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2) &&
        tensor_matches(GetModelOutputTensor_palm(3), kTfLiteInt8, keypoints,
                       PALM_OUTPUT_SCALE_PARTITIONEDCALL_1, PALM_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1) &&
        tensor_matches(GetModelInputTensor_landmark(), kTfLiteInt8, landmark_input,
                       LANDMARK_INPUT_SCALE_SERVING_DEFAULT_INPUT_0, LANDMARK_INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_0) &&
        tensor_matches(GetModelOutputTensor_landmark(0), kTfLiteInt8, scalar,
                       LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_1, LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_1) &&
        tensor_matches(GetModelOutputTensor_landmark(1), kTfLiteInt8, xyz,
                       LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_0, LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_0) &&
        tensor_matches(GetModelOutputTensor_landmark(2), kTfLiteInt8, scalar,
                       LANDMARK_OUTPUT_SCALE_PARTITIONEDCALL_2, LANDMARK_OUTPUT_ZERO_POINT_PARTITIONEDCALL_2) &&
        tensor_matches(GetModelInputTensor_gesture(), kTfLiteFloat32, gesture_input) &&
        tensor_matches(GetModelOutputTensor_gesture(0), kTfLiteFloat32, gesture_output);
    if (!valid) {
        error("TFLM model input/output contract does not match this application.");
        return -1;
    }
    initialized = true;
    return 0;
}
