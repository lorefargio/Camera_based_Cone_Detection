#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// --- PREPROCESSING ---
template <typename T>
__global__ void preprocess_kernel_optimized(const uint8_t* src, T* dst, int src_w, int src_h, int dst_w, int dst_h, int channels) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        float sx = (float)dx * (src_w - 1) / (dst_w - 1);
        float sy = (float)dy * (src_h - 1) / (dst_h - 1);

        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);

        float dx_weight = sx - x0;
        float dy_weight = sy - y0;

        int area = dst_w * dst_h;

        for (int c = 0; c < 3; ++c) {
            int channel_idx = 2 - c; 
            float p00 = src[(y0 * src_w + x0) * channels + channel_idx];
            float p01 = src[(y0 * src_w + x1) * channels + channel_idx];
            float p10 = src[(y1 * src_w + x0) * channels + channel_idx];
            float p11 = src[(y1 * src_w + x1) * channels + channel_idx];

            float val = (1.0f - dx_weight) * (1.0f - dy_weight) * p00 +
                        dx_weight * (1.0f - dy_weight) * p01 +
                        (1.0f - dx_weight) * dy_weight * p10 +
                        dx_weight * dy_weight * p11;

            dst[c * area + dy * dst_w + dx] = (T)(val / 255.0f);
        }
    }
}

// --- REFORMAT PROTOTYPES (CHW to HWC) ---
template <typename T>
__global__ void reformat_prototypes_kernel(const T* src, T* dst, int w, int h, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        for (int c = 0; c < channels; ++c) {
            dst[(y * w + x) * channels + c] = src[c * (w * h) + (y * w + x)];
        }
    }
}

// --- POSTPROCESSING (Using Reformatted HWC Prototypes) ---
#define MAX_DETECTIONS_SHARED 128

template <typename T>
__global__ void postprocess_mask_kernel_optimized(
    const T* output0,
    const T* output1_hwc, // Input is now HWC
    uint8_t* mask_canvas,
    int canvas_w, int canvas_h,
    float conf_threshold
) {
    __shared__ float s_coeffs[MAX_DETECTIONS_SHARED][32];
    __shared__ float s_bboxes[MAX_DETECTIONS_SHARED][5];
    __shared__ uint8_t s_class_ids[MAX_DETECTIONS_SHARED];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = tid; i < MAX_DETECTIONS_SHARED; i += threads_per_block) {
        const T* row = output0 + i * 38;
        s_bboxes[i][0] = (float)row[0] * canvas_w / 640.0f;
        s_bboxes[i][1] = (float)row[1] * canvas_h / 640.0f;
        s_bboxes[i][2] = (float)row[2] * canvas_w / 640.0f;
        s_bboxes[i][3] = (float)row[3] * canvas_h / 640.0f;
        s_bboxes[i][4] = (float)row[4];
        s_class_ids[i] = (uint8_t)row[5];
        for (int c = 0; c < 32; ++c) s_coeffs[i][c] = (float)row[6 + c];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= canvas_w || y >= canvas_h) return;

    uint8_t best_id = 0;
    float max_conf = conf_threshold;

    int px = x * 160 / canvas_w;
    int py = y * 160 / canvas_h;
    
    // Contiguous access to all 32 channels for this pixel!
    const T* pixel_protos = output1_hwc + (py * 160 + px) * 32;

    for (int i = 0; i < MAX_DETECTIONS_SHARED; ++i) {
        float conf = s_bboxes[i][4];
        if (conf < max_conf) continue;

        if (x >= s_bboxes[i][0] && x <= s_bboxes[i][2] && y >= s_bboxes[i][1] && y <= s_bboxes[i][3]) {
            float mask_logit = 0.0f;
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                mask_logit += s_coeffs[i][c] * (float)pixel_protos[c];
            }
            if (1.0f / (1.0f + expf(-mask_logit)) > 0.5f) {
                best_id = s_class_ids[i] + 1;
                max_conf = conf;
            }
        }
    }
    mask_canvas[y * canvas_w + x] = best_id;
}

extern "C" void launch_preprocess(const uint8_t* src, void* dst, int src_w, int src_h, int dst_w, int dst_h, int channels, bool is_fp16, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    if (is_fp16) preprocess_kernel_optimized<half><<<grid, block, 0, stream>>>(src, (half*)dst, src_w, src_h, dst_w, dst_h, channels);
    else preprocess_kernel_optimized<float><<<grid, block, 0, stream>>>(src, (float*)dst, src_w, src_h, dst_w, dst_h, channels);
}

extern "C" void launch_reformat_prototypes(const void* src, void* dst, int w, int h, int channels, bool is_fp16, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (is_fp16) reformat_prototypes_kernel<half><<<grid, block, 0, stream>>>((half*)src, (half*)dst, w, h, channels);
    else reformat_prototypes_kernel<float><<<grid, block, 0, stream>>>((float*)src, (float*)dst, w, h, channels);
}

extern "C" void launch_postprocess_mask(const void* output0, const void* output1_hwc, uint8_t* mask_canvas, int canvas_w, int canvas_h, float conf_threshold, bool is_fp16, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((canvas_w + block.x - 1) / block.x, (canvas_h + block.y - 1) / block.y);
    if (is_fp16) postprocess_mask_kernel_optimized<half><<<grid, block, 0, stream>>>((half*)output0, (half*)output1_hwc, mask_canvas, canvas_w, canvas_h, conf_threshold);
    else postprocess_mask_kernel_optimized<float><<<grid, block, 0, stream>>>((float*)output0, (float*)output1_hwc, mask_canvas, canvas_w, canvas_h, conf_threshold);
}
