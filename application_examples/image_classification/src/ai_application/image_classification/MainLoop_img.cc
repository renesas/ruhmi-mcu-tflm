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
#include "Classifier.hpp"
#include "ImgClassProcessing.hpp"
#include "log_macros.h"
#include "common_util.h"
#include "Labels.h"

extern "C" {
#include "time_counter.h"
#include <wrapper.h>
void update_classification_result(unsigned index, unsigned short category, float probability, unsigned total_count);
}

#define IMAGE_DATA_SIZE  (AI_INPUT_IMAGE_WIDTH * AI_INPUT_IMAGE_HEIGHT * AI_INPUT_IMAGE_BYTE_PER_PIXEL)
extern int8_t model_buffer_int8[IMAGE_DATA_SIZE];

static bool PresentInferenceResult(const std::vector<arm::app::ClassificationResult> &results)
{
    for (uint32_t i = 0; i < results.size(); ++i) {
        update_classification_result((unsigned)i, (unsigned short)results[i].m_labelIdx, (float)results[i].m_normalisedVal, (unsigned)results.size());
    }

    return true;
}

void copy_data_to_model(int8_t* ptr, const uint8_t* src, size_t size) {
    /* The camera buffer contains raw RGB888 bytes, not quantized int8 values. */
    for(size_t i = 0; i < size; i++){
        ptr[i] = static_cast<int8_t>(static_cast<int>(src[i]) - 128);
    }
}

bool main_loop_image_classification()
{
    copy_data_to_model(model_input_ptr(), reinterpret_cast<const uint8_t*>(model_buffer_int8), model_input_size());
    /* Run inference over this image. */
    const uint32_t old_counter = TimeCounter_CurrentCountGet();
    const bool inference_ok = model_invoke();
    const uint32_t new_counter = TimeCounter_CurrentCountGet();
    application_processing_time.ai_inference_time_ms = TimeCounter_CountValueConvertToMs(old_counter, new_counter);
    if (!inference_ok) {
        error("TFLM inference failed: %d", GetLastInvokeStatus());
        return false;
    }

    TfLiteTensor* outputTensor = model_output_tensor();
    const int label_count = getLabelSize();
    if (label_count <= 0 || outputTensor->bytes != static_cast<size_t>(label_count)) {
        error("Model output size does not match the label count.");
        return false;
    }
    arm::app::Classifier classifier;
    std::vector<arm::app::ClassificationResult> results;
    arm::app::ImgClassPostProcess postProcess = arm::app::ImgClassPostProcess(outputTensor,
            classifier,
            results);

    if (!postProcess.DoPostProcess()) {
        error("Post-processing failed.");
        return false;
    }

    /* Add results to context for access outside handler. */
    if (!PresentInferenceResult(results)) {
        return false;
    }
    return true;
}
