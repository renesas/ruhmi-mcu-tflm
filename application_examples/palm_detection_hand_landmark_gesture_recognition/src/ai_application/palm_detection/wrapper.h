/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MODEL_WRAPPER_H
#define MODEL_WRAPPER_H

#include "../tflm_palm/model_palm.h"
#include <stdint.h>

static inline int8_t* palm_input_ptr(void) {
    return GetModelInputPtr_palm_serving_default_input_0();
}

/* Palm model has 4 outputs: scores, boxes, keypoint0, keypoint2 */
static inline int8_t* palm_output_scores_ptr(void) {
    return GetModelOutputPtr_palm_PartitionedCall_3();
}

static inline int8_t* palm_output_boxes_ptr(void) {
    return GetModelOutputPtr_palm_PartitionedCall_0();
}

static inline int8_t* palm_output_kp0_ptr(void) {
    return GetModelOutputPtr_palm_PartitionedCall_2();
}

static inline int8_t* palm_output_kp2_ptr(void) {
    return GetModelOutputPtr_palm_PartitionedCall_1();
}

static inline int palm_invoke(void) {
    RunModel_palm(false);
    return GetLastInvokeStatus_palm();
}

#endif // MODEL_WRAPPER_H
