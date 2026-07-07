#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// 輔助函式：將整數向上取整到下一個 2 的冪次
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// 在 global memory 陣列 `input` 上實作 exclusive scan，
// 並將結果寫入 global memory 的 `result`。
//
// N 是輸入與輸出陣列的邏輯大小，不過
// 學生可以假設 start 與 result 陣列都已依 cudaScan() 註釋所述，
// 以 2 的冪次大小配置。這很有幫助，因為你的 parallel scan
// 很可能會寫到 N 以外的記憶體位置，但當然不會超過
// N 向上取整到下一個 2 的冪次的大小。
//
// 此外，依 cudaScan() 的註釋，你可以實作
// 「原地（in-place）」scan，因為計時框架會複製 input
// 並放到 result 中。
__global__ void upsweep_kernel(int* data, int twod, int N){
    int twod1 = twod * 2;
    // k 是「第幾個工作」，不是陣列 index
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k * twod1;  // 對應 CPU loop的 i

    if (i < N){
        data[i + twod1 - 1] += data[i + twod - 1];
    }
}
__global__ void downsweep_kernel(int* data, int twod, int N){
    int twod1 = twod * 2;

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k * twod1;

    if (i < N){
        int tmp = data[i + twod - 1];
        data[i + twod - 1] = data[i + twod1 - 1];
        data[i + twod1 - 1] = tmp + data[i + twod1 - 1];
    }
}
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO：
    //
    // 在此實作你的 exclusive scan。請記住，
    // 雖然此函式的參數是 device 上配置的陣列，
    // 但這個函式本身是在 CPU 的執行緒上執行。
    // 你的實作需要多次呼叫 CUDA kernel 函式（需自行撰寫）
    // 來完成 scan。
    int rounded_N = nextPow2(N);
    cudaMemcpy(result, input, sizeof(int)*N, cudaMemcpyDeviceToDevice);
    if (rounded_N > N) {
        cudaMemset(result + N, 0, sizeof(int) * (rounded_N - N));
    }
    for (int twod = 1; twod <= rounded_N / 2; twod*=2){
        int twod1 = twod * 2;
        int num_threads = rounded_N / twod1;
        int blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twod, rounded_N);
    }
    // 設result[N-1]為0
    cudaMemset(result + (rounded_N - 1), 0, sizeof(int));

    for(int twod = rounded_N / 2; twod >= 1; twod /=2){
        int twod1 = twod * 2;
        int num_threads = rounded_N / twod1;
        int blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twod, rounded_N);
    }

}


//
// cudaScan --
//
// 這是包在學生 scan 實作外層的計時 wrapper——
// 它會把輸入複製到 GPU，並計量 exclusive_scan() 的執行時間。
// 學生不應修改此函式。
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // 這段程式會傳給 exclusive_scan 的陣列長度向上取整到 2 的冪次，
    // 但原始輸入結尾之後的元素不會被初始化，也不會檢查正確性。
    //
    // 學生的 exclusive_scan 實作可以為了簡化，假設陣列配置長度是 2 的冪次。
    // 這會在輸入長度不是 2 的冪次時多做額外工作，但換取只處理 2 的冪次的簡單實作是值得的。

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // 為了方便，device 上的 input 與 output 向量都初始化成輸入值。
    // 這表示學生若需要，可以在 result 向量上實作 in-place scan。
    // 若這樣做，從 find_repeats 呼叫 exclusive_scan 時要記住這一點。
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // 等待 GPU 完成
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// 包在 Thrust 函式庫 exclusive scan 外層的 wrapper。
// 與上面的 cudaScan() 相同，此函式會把輸入複製到 GPU，
// 且只計量 scan 本身的執行時間。
//
// 不要求學生的實作效能能與 Thrust 版本競爭，但挑戰看看也很有趣。
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// 給定整數陣列 `device_input`，回傳所有滿足
// `device_input[i] == device_input[i+1]` 的 index `i` 所組成的陣列。
//
// 把「找 repeat」變成「找 flag=1 的位置」
__global__ void mark_repeats_kernel(int* input, int* flags, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length - 1){
        flags[i] = (input[i] == input[i + 1]) ? 1 : 0;
    } else if (i < length) {
        // 最後一個元素沒有 i+1 可比，一定不是 repeat 起點
        flags[i] = 0;
    }
}
__global__ void scatter_repeats_kernel(int* flags, int* scanned, int* output, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length - 1 && flags[i] == 1) {
        output[scanned[i]] = i;
    }
}
// 回傳找到的配對總數
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO：
    //
    // 實作此函式。你可能需要
    // 呼叫一次或多次 exclusive_scan()，
    // 以及額外啟動 CUDA kernel。
    //
    // 注意：與 scan 程式相同，呼叫端會確保
    // 配置的陣列大小是 2 的冪次，因此你可以對它們使用 exclusive_scan。
    // 不過，你的實作必須在給定實際陣列長度時，
    // 仍確保 find_repeats 的結果正確。
    int rounded_length = nextPow2(length);
    int* flags = nullptr;
    int* scanned = nullptr;
    cudaMalloc((void**)&flags, rounded_length * sizeof(int));
    cudaMalloc((void**)&scanned, rounded_length * sizeof(int));

    int blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mark_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, flags, length);

    if (rounded_length > length){
        cudaMemset(flags + length, 0, sizeof(int) * (rounded_length - length));
    }
    // --- Step 2：對 flags 做 exclusive scan ---
    exclusive_scan(flags, length, scanned);

    // --- Step 3：依 scan 結果 scatter index 到 output
    scatter_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(flags, scanned, device_output, length);

    // --- Step 4：計算 repeat 總數
    int total_repeats = 0;
    cudaMemcpy(&total_repeats, scanned + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(flags);
    cudaFree(scanned);


    return total_repeats; 
}


//
// cudaFindRepeats --
//
// 包在 find_repeats 外層的計時 wrapper。你不應修改此函式。
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // 設定輸出數量與結果陣列
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
