/*
 * landmark_wrapper.h
 *
 * Thin inline wrappers for the hand-landmark TFLM model, analogous to
 * wrapper.h for the palm detection model.
 *
 * The landmark model has:
 *   - 1 input:  serving_default_input_0  [1,224,224,3] INT8 NHWC
 *   - 3 outputs:
 *       PartitionedCall_0_70174  [1,63]  XYZ landmarks  (scale=1.1077, zp=-43)
 *       PartitionedCall_1_70170  [1, 1]  hand score     (scale=0.003906, zp=-128)
 *       PartitionedCall_2_70172  [1, 1]  handedness     (scale=0.003906, zp=-128)
 */
#ifndef LANDMARK_WRAPPER_H
#define LANDMARK_WRAPPER_H

#include "../tflm_landmark/model_landmark.h"
#include <stdint.h>
#include <stdbool.h>

/* --- Input -------------------------------------------------------- */
static inline int8_t* landmark_input_ptr(void) {
    return GetModelInputPtr_landmark_serving_default_input_0();
}

/* --- Outputs ------------------------------------------------------ */
/* XYZ landmarks [63] */
static inline int8_t* landmark_output_xyz_ptr(void) {
    return GetModelOutputPtr_landmark_PartitionedCall_0();
}

/* Hand score [1] */
static inline int8_t* landmark_output_score_ptr(void) {
    return GetModelOutputPtr_landmark_PartitionedCall_1();
}

/* Handedness [1] */
static inline int8_t* landmark_output_hand_ptr(void) {
    return GetModelOutputPtr_landmark_PartitionedCall_2();
}

/* --- Invoke ------------------------------------------------------- */
static inline int landmark_invoke(void) {
    RunModel_landmark(false);
    return GetLastInvokeStatus_landmark();
}

#endif /* LANDMARK_WRAPPER_H */
