#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// 回傳 GB/sec
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


// 這是會在 GPU 上執行的 CUDA kernel 函式。
// 你可以從它被標記為 __global__ 來判斷。
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // 計算這個 thread 在整個網格中的總索引。
    // 在這個範例中只需要一維計算，所以只看 blockDim.x 和 threadIdx.x。
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // 這個檢查是必要的，因為 N 可能不是 thread block 大小的倍數。
    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


// saxpyCuda --
//
// 這個函式是 CPU 上執行的常規 C 程式碼。
// 它會使用 CUDA API 在 GPU 上配置記憶體、
// 將 CPU 記憶體中的資料搬移到 GPU、
// 並啟動 GPU 上的 kernel 函式。
void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    // 這個函式必須讀取兩個輸入陣列 (xarray 和 yarray)，
    // 並把結果寫入輸出陣列 (resultarray)。
    int totalBytes = sizeof(float) * 3 * N;

    // 計算 block 數量與每個 block 的 thread 數。
    // 在這個範例中，我們把每個 thread block 固定為 512 個 CUDA thread。
    const int threadsPerBlock = 512;

    // 這裡做向上取整，確保能為每個元素分配一個 thread。
    // 這段程式可以處理 N 不是 threadPerBlock 倍數的情況。
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 這些指標將指向在 GPU 上配置的記憶體。
    // 你應該使用 cudaMalloc 來分配它們。
    // 這些 buffer 可以在 CUDA device kernel 程式中存取，
    // 但這個 CPU 執行緒無法直接讀寫 GPU 記憶體中的內容。
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_result = nullptr;
    
    //
    // CS149 TODO：使用 cudaMalloc 在 GPU 上配置 device memory buffer。
    //
    // 我們非常推薦你看 NVIDIA 的教學，
    // 這裡有非常清楚的範例可以一步一步跟著做：
    //
    // https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
    //
    cudaMalloc(&device_x, sizeof(float) * N);
    cudaMalloc(&device_y, sizeof(float) * N);
    cudaMalloc(&device_result, sizeof(float) * N);
    // 在配置完 device memory 後開始計時。
    double startTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO：使用 cudaMemcpy 將輸入陣列複製到 GPU。
    //
    cudaMemcpy(device_x, xarray, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, sizeof(float) * N, cudaMemcpyHostToDevice);

   
    // 啟動 CUDA kernel。
    // 這裡的 <<< >>> 表示這是 CUDA kernel launch。
    // GPU 的實際計算會在這一行發生。
    double kernelStartTime = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();
    double kernelEndTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO：使用 cudaMemcpy 將結果從 GPU 複製回 CPU。
    //
    cudaMemcpy(resultarray, device_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    // 在結果已經複製回 host memory 之後結束計時。
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Effective BW by CUDA kernel only: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, GBPerSec(totalBytes, kernelDuration));

    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));

    //
    // CS149 TODO：使用 cudaFree 釋放 GPU 上的記憶體 buffer。
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void printCudaInfo() {

    // 印出這台機器上的 GPU 統計資訊。
    // 如果你想知道自己跑在哪張 GPU 上，這很有用。

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
