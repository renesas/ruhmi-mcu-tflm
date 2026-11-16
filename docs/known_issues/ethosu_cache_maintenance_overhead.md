# FSP 6.6: Redundant Ethos-U Cache Maintenance Increases NPU Inference Time

## Summary and Status

On FSP 6.6, CPU/NPU applications spend avoidable time in data cache maintenance around every Ethos-U inference. Results stay correct, but inference is slower. The effect is largest for small models, where cache maintenance makes up a larger share of total runtime.

FSP 6.6.0 includes an updated Ethos-U driver. In earlier versions, the driver chose which regions needed cache maintenance. In practice, it maintained only the tensor arena. That was sufficient because input and output tensors are allocated in the arena. The new driver passes every base region to the cache maintenance hooks and leaves region selection to the user implementation. The FSP porting code was not updated for this change. As a result, it cleans and invalidates every region, including overlapping and read-only regions.

## Relevant Environment

| Item | Configuration |
|------|---------------|
| Target | EK-RA8P1 (Cortex-M85 @ 1 GHz, Ethos-U55-256 @ 500 MHz) |
| FSP integration | 6.6.0 (Ethos-U core driver 1.0.0) |
| Previous environment | FSP 6.4 (Ethos-U core driver 0.16) |
| Vela | 5.2.0, Performance mode |
| Affected application variants | CPU/NPU, with D-cache enabled |


## Known Problem

The driver calls two hooks for each inference. Their default implementations in `ethosu_driver.c` are weak, and `rm_ethosu.c` provides strong replacements:

| Hook | Called from | FSP 6.6 `rm_ethosu.c` behavior |
|------|-------------|--------------------------------|
| `ethosu_flush_dcache()` | `ethosu_invoke_async()`, before the NPU starts | `SCB_CleanDCache_by_Addr()` over **every** base region |
| `ethosu_invalidate_dcache()` | `ethosu_wait()`, after the NPU completes | `SCB_CleanInvalidateDCache_by_Addr()` over **every** base region |

If `base_addr` is `NULL`, both hooks operate on the whole cache.

Driver behavior changed between the two FSP versions:

| | FSP 6.4 (driver 0.16) | FSP 6.6 (driver 1.0.0) |
|---|---|---|
| Region selection | Driver mask; by default only the scratch region (base 1, the arena) | All base regions are passed to the hook |
| Who decides what to maintain | Driver | Hook implementation (user / FSP port) |
| FSP port adapted? | — | No; it maintains every region it receives |

The typical base regions for a TFLM + Vela model are:

| Base | Content | Why maintaining it is redundant |
|------|---------|---------------------------------|
| 0 | Weights / command stream | Read-only for both the CPU and the NPU. When it is in MRAM/flash, the CPU never writes it. |
| 1 | Tensor arena (scratch) | **Required.** This is the only region the CPU and NPU share for writing. |
| 2 | Fast scratch | With the default single-arena placement, it aliases the arena, so the same lines are maintained twice. |
| 3+ | Input/output tensors | Located inside the arena and already covered by base 1. |

The cost of address-based maintenance scales with region size. It is paid twice per inference, once before the NPU starts and once after it completes. Therefore, small models with short NPU runtime are affected most.

## Scope and Symptoms

- Outputs are unchanged. Only inference time increases.
- The overhead is visible as a gap between FSP 6.4 and FSP 6.6 timings.
- The overhead does not come from TFLM or the compiler. In the tests below, the TFLM version, compiler, model, and Vela output stay the same. Only the hook implementation changes.
- Disabling D-cache (`BSP_CFG_DCACHE_ENABLED = 0`) or removing cache maintenance is **not** an acceptable workaround. It either slows the CPU or risks stale data.

Measured impact (CPU cycles per inference, Vela 5.2 Performance, same firmware, stock FSP hooks vs. workaround):

| Model | FSP 6.6 stock | With workaround | Change |
|-------|--------------:|----------------:|-------:|
| Auto Encoder | 228,833 | 132,633 | −42.0% |
| RNNoise | 119,620 | 72,001 | −39.8% |
| MCUNet | 1,115,024 | 746,388 | −33.1% |
| ResNet-8 | 712,760 | 571,484 | −19.8% |
| MobileNet V3 | 7,279,299 | 6,052,372 | −16.9% |
| MobileNet V1 | 2,945,763 | 2,498,148 | −15.2% |
| Tiny2Waveletter | 11,442,939 | 9,706,720 | −15.2% |
| Visual Wake Word | 527,160 | 449,252 | −14.8% |
| MicroNet | 2,292,044 | 1,963,711 | −14.3% |
| Keyword Spotting | 140,381 | 120,312 | −14.3% |
| MobileNet V2 | 18,035,223 | 15,824,176 | −12.3% |
| PoseNet | 5,983,301 | 5,283,836 | −11.7% |
| SqueezeNet | 9,308,700 | 8,280,104 | −11.0% |
| MobileFaceNet | 8,813,208 | 8,076,108 | −8.4% |
| BlazeFace | 3,663,209 | 3,395,861 | −7.3% |
| MediaPipe | 7,854,844 | 7,309,596 | −6.9% |
| YOLOX Tiny | 43,436,137 | 40,975,680 | −5.7% |
| NanoDet | 117,792,520 | 111,534,964 | −5.3% |
| Yolo-fastest | 18,957,584 | 18,133,172 | −4.3% |
| ShuffleNet | 27,560,883 | 26,672,280 | −3.2% |

## Proposed Workaround

Replace the hooks with a policy that forwards only the regions that need maintenance to the unmodified FSP implementation:

- Skip base 0 (read-only weights).
- Skip regions with size 0.
- Skip regions fully contained in an earlier region with index 1 or higher, such as the fast-scratch alias and IO tensors inside the arena.
- Pass each remaining region to the original FSP function separately.
- Forward `base_addr == NULL` (whole cache) unchanged.

This reduces redundant maintenance. It does **not** disable cache maintenance.

> **Limitation:** the policy assumes that base 0 is read-only for the CPU after startup. If the application copies weights from flash to RAM, call `SCB_CleanDCache_by_Addr()` on the destination **once** after the copy and before the first inference. Re-check the policy if a project places IO tensors outside the arena or uses a custom Vela memory layout.

<details>
<summary>Manual workaround: source files and project setting</summary>

### 1. Add `src/ethosu_fsp_cache_adapter.c`

This file compiles the unmodified FSP module once and renames its hooks so that the policy can call them.

```c
/*
 * Compile the unmodified FSP module once, with private names for its cache hooks.
 * Exclude ra/fsp/src/rm_ethosu/rm_ethosu.c from standalone compilation.
 */
#define ethosu_flush_dcache      benchmark_fsp_flush_dcache
#define ethosu_invalidate_dcache benchmark_fsp_invalidate_dcache
#include "../ra/fsp/src/rm_ethosu/rm_ethosu.c"
#undef ethosu_flush_dcache
#undef ethosu_invalidate_dcache
```

### 2. Add `src/ethosu_cache_policy.c`

```c
#include "hal_data.h"

void benchmark_fsp_flush_dcache(const uint64_t * base_addr, const size_t * base_addr_size, int num_base_addr);
void benchmark_fsp_invalidate_dcache(const uint64_t * base_addr, const size_t * base_addr_size, int num_base_addr);

/* Read after inference to confirm the policy actually ran. */
volatile uint32_t g_ethosu_cache_policy_calls = 0;

static bool ethosu_region_needed (const uint64_t * base_addr, const size_t * base_addr_size, int i)
{
    if ((0 == i) || (0U == base_addr_size[i]))
    {
        return false;
    }

    /* Skip regions fully covered by an earlier region (fast scratch alias, IO tensors in arena). */
    for (int j = 1; j < i; j++)
    {
        if ((base_addr[i] >= base_addr[j]) &&
            ((base_addr[i] + base_addr_size[i]) <= (base_addr[j] + base_addr_size[j])))
        {
            return false;
        }
    }

    return true;
}

__attribute__((used, retain)) void ethosu_flush_dcache (const uint64_t * base_addr, const size_t * base_addr_size, int num_base_addr)
{
    g_ethosu_cache_policy_calls++;
    if (NULL == base_addr)
    {
        benchmark_fsp_flush_dcache(base_addr, base_addr_size, num_base_addr);
        return;
    }

    for (int i = 0; i < num_base_addr; i++)
    {
        if (ethosu_region_needed(base_addr, base_addr_size, i))
        {
            benchmark_fsp_flush_dcache(&base_addr[i], &base_addr_size[i], 1);
        }
    }
}

__attribute__((used, retain)) void ethosu_invalidate_dcache (const uint64_t * base_addr, const size_t * base_addr_size, int num_base_addr)
{
    g_ethosu_cache_policy_calls++;
    if (NULL == base_addr)
    {
        benchmark_fsp_invalidate_dcache(base_addr, base_addr_size, num_base_addr);
        return;
    }

    for (int i = 0; i < num_base_addr; i++)
    {
        if (ethosu_region_needed(base_addr, base_addr_size, i))
        {
            benchmark_fsp_invalidate_dcache(&base_addr[i], &base_addr_size[i], 1);
        }
    }
}
```

### 3. Exclude the Original `rm_ethosu.c` from the Build

In e² studio, right-click `ra/fsp/src/rm_ethosu/rm_ethosu.c` and select **Resource Configurations → Exclude from Build...**. Select every build configuration (Debug and Release). This change is stored in `.cproject`, not in the FSP configuration (`configuration.xml`):

```xml
<sourceEntries>
    <entry excluding="fsp/src/rm_ethosu/rm_ethosu.c" flags="VALUE_WORKSPACE_PATH|RESOLVED" kind="sourcePath" name="ra"/>
    ...
</sourceEntries>
```

</details>

### Check That the Workaround Is Active

Misconfiguration fails at link time instead of silently falling back:

| Mistake | Result |
|---------|--------|
| Adapter file missing | Undefined reference to `benchmark_fsp_flush_dcache` / `benchmark_fsp_invalidate_dcache` |
| `rm_ethosu.c` not excluded (for example, after regeneration) | Duplicate symbol `ethosu_flush_dcache` / `ethosu_invalidate_dcache` |

At runtime, read `g_ethosu_cache_policy_calls` after an inference. It must be greater than zero, normally two per inference. A value of zero means that the policy is not linked in.

## Target-Side Validation

| Check | Scope | Result |
|-------|-------|--------|
| Output equivalence, stock vs. workaround | 20 models; 12 input pairs per model | All output bytes identical |
| Performance | 20 models | −3.2% to −42.0% |

The output tests used zero, extreme, alternating, ramp and random input patterns, including consecutive inferences and RNNoise state feedback. MCUNet, RNNoise and NanoDet were tested with an earlier implementation of the same cache filtering policy; the other 17 models used the workaround above. The tested layouts used an SRAM arena and weights in MRAM or SDRAM.

These tests compare stock and optimized outputs for the tested inputs; they do not establish accuracy against a reference or cover every memory layout.
