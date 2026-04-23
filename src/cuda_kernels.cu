#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// Preprocessing Kernel: BGR/RGBA to RGB, Resize, Normalize, HWC to CHW
__global__ void preprocess_kernel(const uint8_t* src, float* dst, int src_w, int src_h, int dst_w, int dst_h, int channels) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        float sx = (float)dx * src_w / dst_w;
        float sy = (float)dy * src_h / dst_h;

        int x0 = (int)sx;
        int y0 = (int)sy;
        
        // Simple nearest neighbor for brevity (can be upgraded to bilinear)
        int src_idx = (y0 * src_w + x0) * channels;
        
        // Output is CHW
        int area = dst_w * dst_h;
        dst[0 * area + dy * dst_w + dx] = src[src_idx + 2] / 255.0f; // R
        dst[1 * area + dy * dst_w + dx] = src[src_idx + 1] / 255.0f; // G
        dst[2 * area + dy * dst_w + dx] = src[src_idx + 0] / 255.0f; // B
    }
}

// Postprocessing: Mask Canvas Generation Kernel (Winner-take-all based on confidence)
__global__ void postprocess_mask_kernel(
    const float* output0,  // Detections (300, 38) -> [x1, y1, x2, y2, conf, class, mask_coeffs...]
    const float* output1,  // Prototypes (32, 160, 160)
    uint8_t* mask_canvas,  // Output ID map (W, H)
    int canvas_w, int canvas_h,
    float conf_threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= canvas_w || y >= canvas_h) return;

    uint8_t best_id = 0;
    float max_conf = conf_threshold;

    // Scaling factors for prototypes (YOLO segmentation standard is 1/4 of input)
    // Assuming input 640x640, prototypes are 160x160
    int px = x * 160 / canvas_w;
    int py = y * 160 / canvas_h;

    for (int i = 0; i < 300; ++i) {
        const float* row = output0 + i * 38;
        float conf = row[4];
        if (conf < max_conf) continue;

        // BBox check (scaled to canvas)
        float x1 = row[0] * canvas_w / 640.0f;
        float y1 = row[1] * canvas_h / 640.0f;
        float x2 = row[2] * canvas_w / 640.0f;
        float y2 = row[3] * canvas_h / 640.0f;

        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
            // Compute linear combination of prototypes and coefficients
            float mask_logit = 0.0f;
            for (int c = 0; c < 32; ++c) {
                mask_logit += row[6 + c] * output1[c * 160 * 160 + py * 160 + px];
            }
            
            // Fast sigmoid thresholding
            if (1.0f / (1.0f + expf(-mask_logit)) > 0.5f) {
                best_id = (uint8_t)(i + 1);
                max_conf = conf;
            }
        }
    }
    mask_canvas[y * canvas_w + x] = best_id;
}

// Wrapper functions for C++ integration
extern "C" void launch_preprocess(const uint8_t* src, float* dst, int src_w, int src_h, int dst_w, int dst_h, int channels, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    preprocess_kernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h, channels);
}

extern "C" void launch_postprocess_mask(const float* output0, const float* output1, uint8_t* mask_canvas, int canvas_w, int canvas_h, float conf_threshold, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((canvas_w + block.x - 1) / block.x, (canvas_h + block.y - 1) / block.y);
    postprocess_mask_kernel<<<grid, block, 0, stream>>>(output0, output1, mask_canvas, canvas_w, canvas_h, conf_threshold);
}
