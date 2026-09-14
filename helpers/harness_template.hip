/*
 * Standalone HIP profiling harness for DCU (gfx938 / DTK).
 *
 * Build (inside the DTK container):
 *   hipcc --offload-arch=gfx938 -O3 harness.hip -o harness
 *
 * Profile:
 *   rocprof -i prof_input.txt -o results.csv ./harness
 *
 * Paste your kernel below, set N/THREADS, fill run_correctness().
 * DTK port notes:
 *   - warp shuffle: use __shfl_xor(v, offset, 32)  (NOT __shfl_xor_sync —
 *     64-bit-mask sync path VMFaults on DTK 26.04, see reference/09).
 *   - __float2bfloat16_rn -> __float2bfloat16, include <hip/hip_bf16.h>.
 *   - no inline PTX; plain vector stores compile to 128-bit global stores.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t e_ = (expr);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Paste your kernel here. Example placeholder:
// ---------------------------------------------------------------------------
__global__ void my_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// Warp reduction that works on DTK (see reference/09-common-issues.md):
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset, 32);   // unmasked intrinsic, explicit width
    return v;
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
static constexpr int N = 1 << 20;        // problem size
static constexpr int THREADS = 256;      // CTA size

int main() {
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_in, 0x3c, N * sizeof(float)));  // ~0.01f pattern

    int blocks = (N + THREADS - 1) / THREADS;

    // -- correctness first: a harness that computes garbage profiles garbage
    my_kernel<<<blocks, THREADS>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());
    float host = 0.f;
    HIP_CHECK(hipMemcpy(&host, d_out, sizeof(float), hipMemcpyDeviceToHost));
    printf("out[0]=%f (expect ~%f)\n", host, 0.011486f * 2.0f);  // 0x3c3c3c3c as float

    // -- timing (never take durations from a profiled run)
    const int WARMUP = 20, ITERS = 200, REPS = 3;
    for (int i = 0; i < WARMUP; i++) my_kernel<<<blocks, THREADS>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < REPS; r++) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < ITERS; i++) my_kernel<<<blocks, THREADS>>>(d_in, d_out, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    printf("kernel: %.2f us (min of %d reps x %d iters)\n",
           best * 1000.0f / ITERS, REPS, ITERS);
    printf("grid: %d CTAs x %d threads = %d work-items\n",
           blocks, THREADS, blocks * THREADS);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
