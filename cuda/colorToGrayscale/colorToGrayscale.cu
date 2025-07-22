#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <cmath>

typedef struct __attribute__((packed)) bitmapfileheader{
    uint8_t bfType[2];
    uint32_t bfSize;
    uint32_t bfReserved;
    uint32_t bfOffBits;
} bitmapfileheader;

typedef struct __attribute__((packed)) bitmapinfoheader{
    uint32_t biSize;
    uint32_t biWidth;
    uint32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    uint32_t biXPelsPerMeter;
    uint32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} bitmapinfoheader;

__global__
void colortoGrayscaleConvertion(uint8_t* imgOut, uint8_t* imgIn, uint32_t width, uint32_t height, uint32_t padding){
    // width and height are in pixels, padding - number of extra bytes per row
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height){
        uint32_t pid = row * width * 3 + row * padding + col * 3;
        uint8_t r = imgIn[pid+0];
        uint8_t g = imgIn[pid+1];
        uint8_t b = imgIn[pid+2];
        uint8_t grayscaleval = 0.21f * r + 0.71f * g + 0.07f * b;
        imgOut[pid] = imgOut[pid+1] = imgOut[pid+2] = grayscaleval; 
    }
}

int main(){
    std::ifstream file("sunset.bmp",std::ios::in | std::ios::binary);
    assert(file.is_open());

    bitmapfileheader bmpHeader;
    bitmapinfoheader bmpInfoHeader;
    file.read((char*)&bmpHeader,14);

    file.read((char*)&bmpInfoHeader,40);
    std::cout << "imgDim: " << bmpInfoHeader.biWidth << " x " << bmpInfoHeader.biHeight << std::endl;
    int sizeinbytes = (uint32_t)ceil(bmpInfoHeader.biWidth/4.0) * 4 * bmpInfoHeader.biHeight * 3;
    uint8_t* img_h = (uint8_t*)malloc(sizeinbytes);
    uint8_t* imgOut_h = (uint8_t*)malloc(sizeinbytes);
    file.read((char*)img_h,sizeinbytes);

    uint8_t* imgIn_d, *imgOut_d;
    cudaMalloc(&imgIn_d,sizeinbytes);
    cudaMalloc(&imgOut_d,sizeinbytes);
    cudaMemcpy(imgIn_d,img_h,sizeinbytes,cudaMemcpyHostToDevice);
    dim3 bd(32,32,1); // block dim
    dim3 gd(ceil(bmpInfoHeader.biWidth/32.0),ceil(bmpInfoHeader.biHeight/32.0),1);
    uint32_t paddingInBytes = (sizeinbytes / bmpInfoHeader.biHeight) - (bmpInfoHeader.biWidth * 3);
    colortoGrayscaleConvertion<<<gd,bd>>>(imgOut_d,imgIn_d,bmpInfoHeader.biWidth,bmpInfoHeader.biHeight,paddingInBytes);

    cudaMemcpy(imgOut_h,imgOut_d,sizeinbytes,cudaMemcpyDeviceToHost);

    std::ofstream outfile("grayset.bmp",std::ios::binary);
    assert(outfile.is_open());
    outfile.write((char *)&bmpHeader,14);
    outfile.write((char *)&bmpInfoHeader,40);
    outfile.write((char *)imgOut_h,sizeinbytes);
    assert(outfile);

    free(img_h); free(imgOut_h);
    cudaFree(imgIn_d); cudaFree(imgOut_d);
    outfile.close();
    file.close();
    return 0;
}