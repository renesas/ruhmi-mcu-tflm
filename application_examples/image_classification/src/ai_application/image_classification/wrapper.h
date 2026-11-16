#ifndef MODEL_WRAPPER_H
#define MODEL_WRAPPER_H

#include "model.h"
#include "ai_application_config.h"

static inline size_t model_input_size(void)
{
    return AI_INPUT_IMAGE_WIDTH * AI_INPUT_IMAGE_HEIGHT * AI_INPUT_IMAGE_BYTE_PER_PIXEL;
}

static inline int model_init(void)
{
    int status = ModelInit();
    if (status != 0)
    {
        return status;
    }

    const TfLiteTensor* input = GetModelInputTensor();
    const TfLiteTensor* output = GetModelOutputTensor();
    if (!input || !output || input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        !input->data.int8 || !output->data.int8 || input->bytes != model_input_size() ||
        !input->dims || input->dims->size != 4 || input->dims->data[0] != 1 ||
        input->dims->data[1] != AI_INPUT_IMAGE_HEIGHT ||
        input->dims->data[2] != AI_INPUT_IMAGE_WIDTH ||
        input->dims->data[3] != AI_INPUT_IMAGE_BYTE_PER_PIXEL ||
        !output->dims || output->dims->size != 2 || output->dims->data[0] != 1 ||
        output->dims->data[1] != 1000 || output->bytes != 1000 ||
        !isfinite(input->params.scale) || !isfinite(output->params.scale) ||
        fabsf(input->params.scale - INPUT_SCALE_SERVING_DEFAULT_INPUT_2_0) > 1e-8f ||
        input->params.zero_point != INPUT_ZERO_POINT_SERVING_DEFAULT_INPUT_2_0 ||
        output->params.scale <= 0.0f ||
        output->quantization.type != kTfLiteAffineQuantization ||
        !output->quantization.params)
    {
        return -3;
    }

    const TfLiteAffineQuantization* quantization =
        (const TfLiteAffineQuantization*) output->quantization.params;
    if (!quantization->scale || quantization->scale->size != 1 ||
        !quantization->zero_point || quantization->zero_point->size != 1 ||
        quantization->quantized_dimension != 0 ||
        !isfinite(quantization->scale->data[0]) ||
        fabsf(quantization->scale->data[0] - output->params.scale) > 1e-8f ||
        quantization->zero_point->data[0] != output->params.zero_point)
    {
        return -3;
    }
    return 0;
}

static inline int8_t* model_input_ptr(void)
{
    return GetModelInputPtr_serving_default_input_2_0();
}

static inline TfLiteTensor* model_output_tensor(void)
{
    return GetModelOutputTensor();
}

static inline bool model_invoke(void)
{
    RunModel(false);
    return GetLastInvokeStatus() == 0;
}

#endif
