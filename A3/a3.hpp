/*  William
 *  Stewart
 *  wlstewar
 */

#ifndef A3_HPP
#define A3_HPP

#include <cmath>


// magic number -- 1/sqrt(2pi)
#define k_scale 0.398942  //(1.0 / sqrt(2.0 * 3.14159))

const int block_size = 64;

template <typename T>
__device__ inline
T k (T x)
{
        return k_scale * exp(-((x * x) / 2.0f));
}


/* Computes K((x-xi)/h) for each i when threads load their
 * element into the local buffer, and then does sequential
 * addressing reduction as discussed in lecture.
 */
template <typename T, int num_threads>
__global__
void f_hat_i (T* ibuf, T* block_buf, T x, T h)
{
        __shared__ T sdata[num_threads];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        sdata[tid] = k((x - ibuf[i]) / h);
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s)
                        sdata[tid] += sdata[tid+s];
                __syncthreads();
        }
        if (tid == 0)
                block_buf[blockIdx.x] = *sdata;
}


/* Reduces the final result from every block into 1 value,
 * and then multiplies by the scaling factor (1/(n*h)).
 */
template <typename T>
__global__ 
void reduce_blocks (T* block_buf, T* obuf, int idx, T scale)
{
        int tid = threadIdx.x;
        extern __shared__ T sdata[];
        sdata[tid] = block_buf[tid];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                        sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
        }
        if (tid == 0)
                obuf[idx] = scale * (*sdata);
}

/* CUDA function to compute gaussian kernel density estimator
 * of given vector and bandwidth.
 */
void gaussian_kde(int n, float h, const std::vector<float>& x, 
                  std::vector<float>& y)
{

        // get number of blocks and get its nearest power of 2 for block reduce
        const int num_blocks = (n + block_size - 1) / block_size;
        const int block_pad  = 1<<((sizeof(int)*8) - __builtin_clz(num_blocks));
        const float f_hat_scale = 1.0f / ((float) n * h);

        // set up gpu memory
        float * d_x = nullptr;
        float * d_y = nullptr;
        float * d_block_buf = nullptr;
        cudaError_t cuda_error;
        cuda_error = cudaMalloc(&d_x, n * sizeof(float));
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not allocate memory for input vector: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                exit(1);
        }
        cuda_error = cudaMalloc(&d_y, n * sizeof(float));
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not allocate memory for output vector: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                cudaFree(d_x);
                exit(1);
        }
        cuda_error = cudaMalloc(&d_block_buf, n * sizeof(float));
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not allocate memory for block buffer: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                cudaFree(d_x);
                cudaFree(d_y);
                exit(1);
        }
        cuda_error = cudaMemcpy(d_x, x.data(), n * sizeof(float),
                                cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not copy input vector to device: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                cudaFree(d_x);
                cudaFree(d_block_buf);
                exit(1);
        }
        cuda_error = cudaMemset(d_block_buf, 0, block_pad  * sizeof(int));
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not initialize block buffer: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                cudaFree(d_x);
                cudaFree(d_block_buf);
                cudaFree(d_y);
                exit(1);
        }

        // run kernel
        for (int i = 0; i < n; ++i) {
                f_hat_i<float, block_size><<<num_blocks, block_size>>>(d_x, d_block_buf, x[i], h);
                reduce_blocks<<<1, block_pad, block_pad>>>(d_block_buf, d_y, i, f_hat_scale);
        }

        // get result
        cuda_error = cudaMemcpy(y.data(), d_y, n * sizeof(float), 
                                cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_error) {
                std::cerr << "could not copy result from gpu memory: "
                        << cudaGetErrorName(cuda_error) << std::endl;
                cudaFree(d_x);
                cudaFree(d_y);
                cudaFree(d_block_buf);
                exit(1);
        }

        // clean up gpu memory
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_block_buf);
} // gaussian_kde
#endif
