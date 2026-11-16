#ifndef TFLM_MEMORY_H
#define TFLM_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include "application_config.h"

#if (TENSOR_ARENA_ALLOCATION == ALLOCATE_TO_ONCHIP_RAM)
#define TFLM_ARENA_ATTRIBUTES __attribute__((aligned(32)))
#elif (TENSOR_ARENA_ALLOCATION == ALLOCATE_TO_SDRAM)
#define TFLM_ARENA_ATTRIBUTES __attribute__((aligned(32), section(".sdram_noinit")))
#else
#error "Unsupported TENSOR_ARENA_ALLOCATION"
#endif

#define TFLM_SHARED_SCRATCH_BYTES 1008640
#define TFLM_PERSISTENT_BYTES 4096

/* Palm and landmark run serially in the AI thread. Persistent state is private. */
extern uint8_t g_tflm_shared_scratch[TFLM_SHARED_SCRATCH_BYTES];

#endif
