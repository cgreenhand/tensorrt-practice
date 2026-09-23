#include<iostream>

struct AffineMatrix {
    float value[6];
};

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

void cuda_preprocess(uint8* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,cudaStream_t stream)
{

    int img_size = src_width * src_height * 3;
    memcpy(img_buffer_host,src,img_size);

    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device,img_buffer_host,img_size,cudaMemcpyHostToDevice,stream));

    AffineMatrix s2d,d2s;

    float scale = std::min(static_cast<float>(dst_width) / src_width, static_cast<float>(dst_height)/ src_height);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = dst_width * 0.5 - scale * src_width * 0.5;

    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = dst_height * 0.5 - scale * src_height * 0.5
    


                



}