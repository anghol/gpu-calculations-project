#include <stdint.h>    /* for uint64 definition */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MILLION 1000000
#define BILLION 1000000000L

#define BLOCK_SIZE 512

#define EPS 0.01


// подынтегральная функция
__host__ __device__ double function(double x)
{   
    return exp(-x*x);
    // return 8 + 2*x - x*x;
}

// ядро - метод Симпсона
__global__ void simpson_kernel(double a, int m, double h, double *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m)
    {
        double x_2i = a + 2*idx * h;
        double x_2i_1 = a + (2*idx + 1) * h;
        double x_2i_2 = a + (2*idx + 2) * h;

        double y_2i = function(x_2i);
        double y_2i_1 = function(x_2i_1);
        double y_2i_2 = function(x_2i_2);

        results[idx] = (h / 3) * (y_2i + 4*y_2i_1 + y_2i_2);
    }
}

// запуск на GPU
uint64_t run_in_gpu(double a, int m, double h)
{
    float dt_gpu;
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // выделение памяти на CPU
    int num_bytes = sizeof(double) * m; 
    double *results = (double *)malloc(num_bytes);

    // указатели на память на видеокарте
    double *results_gpu = NULL;

    // старт события
    cudaEventRecord(start_gpu, 0);

    // выделение памяти на видеокарте
    cudaMalloc((void**)&results_gpu, num_bytes);

    // создание конфигурации потоков и блоков
    dim3 blockSize = dim3(BLOCK_SIZE, 1);
    dim3 numBlocks = dim3(m / blockSize.x, 1);

    // вызов ядра для метода Симпсона
    simpson_kernel<<<numBlocks, blockSize>>>(a, m, h, results_gpu);

    // копирование данных (результат вычислений) с GPU на CPU
    cudaMemcpy(results, results_gpu, num_bytes, cudaMemcpyDeviceToHost);

    // суммирование
    double integral = 0.0;
    for (int i = 0; i < m; ++i) {
        integral += results[i];
    }
    printf("I = %f\n", integral);

    // освобождение видеопамяти
    cudaFree(results_gpu);

    // окончание события и измерение времени выполнения в ns
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&dt_gpu, start_gpu, stop_gpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    dt_gpu *= MILLION;

    // освобождение памяти на CPU
    free(results);

    return dt_gpu;
}

// запуск на CPU
uint64_t run_in_process(double a, int m, double h)
{
    clock_t t0 = clock();
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double integral = 0;
    double x_2i, x_2i_1, x_2i_2;
    double y_2i, y_2i_1, y_2i_2;

    // вычисление значения для каждого из M подотрезков
    for (int i = 0; i < m; ++i) 
    {
        if (i < m)
        {
            x_2i = a + 2*i * h;
            x_2i_1 = a + (2*i + 1) * h;
            x_2i_2 = a + (2*i + 2) * h;

            y_2i = function(x_2i);
            y_2i_1 = function(x_2i_1);
            y_2i_2 = function(x_2i_2);

            integral += (h / 3) * (y_2i + 4*y_2i_1 + y_2i_2);
        }
    }
    printf("I = %f\n", integral);

    clock_gettime(CLOCK_MONOTONIC, &stop);
    uint64_t diff = BILLION * (stop.tv_sec - start.tv_sec) + stop.tv_nsec - start.tv_nsec;
    return diff;
}

int main()
{
    // пределы интегрирования
    double a = -100000;
    double b = 100000; 

    int n_min = (int)((b-a) / EPS);
    int n = (n_min / (BLOCK_SIZE*2)) * (BLOCK_SIZE*2) + (n_min % (BLOCK_SIZE*2)) * (BLOCK_SIZE*2);
    int m = (int)(n / 2);
    double h = (b-a) / n;

    // вычисления на CPU
    uint64_t dt_process = run_in_process(a, m, h);
    printf("Time on CPU: %f s \n", (double) dt_process / BILLION);

    // вычисления на GPU
    uint64_t dt_gpu = run_in_gpu(a, m, h);
    printf("Time on GPU: %f s \n", (double) dt_gpu / BILLION);

    // коэффициент
    printf("Ratio: %f \n", (double)dt_process / (double)dt_gpu);
}