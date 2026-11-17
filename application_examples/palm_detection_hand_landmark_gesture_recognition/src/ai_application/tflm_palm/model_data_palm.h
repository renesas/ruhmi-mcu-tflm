// SPDX-License-Identifier: BSD-3-Clause
// Generated wrapper code (BSD-3-Clause). The model byte array in the
// corresponding .c file retains the original customer model's license.
#ifndef MODEL_DATA_PALM_H
#define MODEL_DATA_PALM_H

#include <stdint.h>
#include <stddef.h>
#include "application_config.h"

/* TFLite model flatbuffer data */
#if (AI_MODEL_ALLOCATION == ALLOCATE_TO_ONCHIP_RAM) || \
    (AI_MODEL_ALLOCATION == ALLOCATE_TO_SDRAM_INITIAL_IN_OSPI) || \
    (AI_MODEL_ALLOCATION == ALLOCATE_TO_ONCHIP_RAM_INITIAL_IN_OSPI)
extern uint8_t g_model_data_palm[];
#else
extern const uint8_t g_model_data_palm[];
#endif
extern const size_t g_model_data_size_palm;

#define MODEL_DATA_SIZE_PALM 1352416

#endif /* MODEL_DATA_PALM_H */
