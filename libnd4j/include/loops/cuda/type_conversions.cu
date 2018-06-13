//
//
//

#include <loops/type_conversions.h>
#include <types/types.h>

namespace nd4j {
    template <typename T>
    void TypeCast::convertToThreshold(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
        nd4j_printf("TypeCast::convertToThreshold: this method shouldn't be ever called\n","");
    }

    template <typename T>
    void TypeCast::convertFromThreshold(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
        nd4j_printf("TypeCast::convertFromThreshold: this method shouldn't be ever called\n","");
    }

    template<typename S, typename T>
    __device__ void convertKernelGeneric(S *x, Nd4jLong N, T *z) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (Nd4jLong i = tid; i < N; i+= blockDim.x * gridDim.x) {
            // despite it's stupid, it simplifies conversion to bottom dtypes
            z[i] = static_cast<T>(static_cast<float>(x[i]));
        }
    };

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 */
    template<typename T>
    __device__ inline void encoderKernelP1Generic(void *dx, Nd4jLong N, void *dz, float threshold) {
        auto x = reinterpret_cast<T *> (dx);
        auto z = reinterpret_cast<int *> (dz);

        //basically, for phase One we want do calculation: how many eligible values we have, and which blocks will be holding data
        Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;

        int pass = tid < N && nd4j::math::nd4j_abs<T>(x[tid]) >= static_cast<T>(threshold) ? 1 : 0;
        int bp=__syncthreads_count(pass);

        if (threadIdx.x == 0) {
            // saving out per-block passes
            z[blockIdx.x+1] = bp;

            // saving out sum
            atomicAdd(&z[0], bp);
        }
    }

    __device__ __inline__ int pow2i (int e){
        return 1<<e;
    }

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif





    template <bool isNP2>
    __device__ void loadSharedChunkFromMem(int *s_data, const int *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB) {
        int thid = threadIdx.x;
        mem_ai = baseIndex + threadIdx.x;
        mem_bi = mem_ai + blockDim.x;

        ai = thid;
        bi = thid + blockDim.x;

        // compute spacing to avoid bank conflicts
        bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        // Cache the computational window in shared memory
        // pad values beyond n with zeros
        s_data[ai + bankOffsetA] = g_idata[mem_ai];

        if (isNP2) { // compile-time decision
            s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
        } else {
            s_data[bi + bankOffsetB] = g_idata[mem_bi];
        }
    }

    template <bool isNP2>
    __device__ void storeSharedChunkToMem(int* g_odata, int* s_data, int n, int ai, int bi, int mem_ai, int mem_bi, int bankOffsetA, int bankOffsetB) {
        __syncthreads();

        // write results to global memory
        g_odata[mem_ai] = s_data[ai + bankOffsetA];
        if (isNP2) { // compile-time decision
            if (bi < n)
                g_odata[mem_bi] = s_data[bi + bankOffsetB];
        } else {
            g_odata[mem_bi] = s_data[bi + bankOffsetB];
        }
    }

    template <bool storeSum>
    __device__ void clearLastElement(int* s_data, int *g_blockSums, int blockIndex) {
        if (threadIdx.x == 0)
        {
            int index = (blockDim.x << 1) - 1;
            index += CONFLICT_FREE_OFFSET(index);

            if (storeSum) { // compile-time decision
                // write this block's total sum to the corresponding index in the blockSums array
                g_blockSums[blockIndex] = s_data[index];
            }

            // zero the last element in the scan so it will propagate back to the front
            s_data[index] = 0;
        }
    }



    __device__ unsigned int buildSum(int *s_data) {
        unsigned int thid = threadIdx.x;
        unsigned int stride = 1;

        // build the sum in place up the tree
        for (int d = blockDim.x; d > 0; d >>= 1) {
            __syncthreads();

            if (thid < d) {
                int i  = __mul24(__mul24(2, stride), thid);
                int ai = i + stride - 1;
                int bi = ai + stride;

                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                s_data[bi] += s_data[ai];
            }

            stride *= 2;
        }

        return stride;
    }

    __device__ void scanRootToLeaves(int *s_data, unsigned int stride) {
        unsigned int thid = threadIdx.x;

        // traverse down the tree building the scan in place
        for (int d = 1; d <= blockDim.x; d *= 2) {
            stride >>= 1;

            __syncthreads();

            if (thid < d) {
                int i  = __mul24(__mul24(2, stride), thid);
                int ai = i + stride - 1;
                int bi = ai + stride;

                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                float t  = s_data[ai];
                s_data[ai] = s_data[bi];
                s_data[bi] += t;
            }
        }
    }

    template <bool storeSum>
    __device__ void prescanBlock(int *data, int blockIndex, int *blockSums) {
        int stride = buildSum(data);               // build the sum in place up the tree
        clearLastElement<storeSum>(data, blockSums,
                                   (blockIndex == 0) ? blockIdx.x : blockIndex);
        scanRootToLeaves(data, stride);            // traverse down tree to build the scan
    }


    template <bool storeSum, bool isNP2>
    __global__ void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
        int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
        extern __shared__ int s_data[];

        // load data into shared memory
        loadSharedChunkFromMem<isNP2>(reinterpret_cast<int *>(s_data), g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);

        // scan the data in each block
        prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);

        // write results to device memory
        storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
    }


    __global__ void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex) {
        __shared__ float uni;
        if (threadIdx.x == 0)
            uni = uniforms[blockIdx.x + blockOffset];

        unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

        __syncthreads();

        // note two adds per thread
        g_data[address] += uni;
        g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
    }

/*
 * This kernel does prefix sum in parallel, to calculate offsets for each block
 */
    template<typename T>
    __device__ inline void encoderKernelP2Generic(void *dx, Nd4jLong n, void *dz) {
        // TODO: to be remove
    }

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 *
 * Based on: https://github.com/knotman90/cuStreamComp <-- efficient CUDA stream compaction algorithm
 */
    template<typename T>
    __device__ inline void encoderKernelP3Generic(void *dx, int *offsets, Nd4jLong N, void *dz) {
        T *x = reinterpret_cast<T *> (dx);
        int *z = reinterpret_cast<int *> (dz);

        Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ int warpTotals[];

        // fetch block offset only once
        __shared__ float threshold;
        __shared__ FloatBits fb;
        __shared__ int bo;
        __shared__ int limit;
        if (threadIdx.x == 0) {
            limit = z[0];
            fb.i_ = z[2];
            threshold = fb.f_;
            bo = offsets[blockIdx.x];
        }
        __syncthreads();

        if (tid < N) {
            T value = x[tid];
            int pred = nd4j::math::nd4j_abs<T>(value) >= static_cast<T>(threshold) ? 1 : 0;
            int w_i = threadIdx.x/warpSize; //warp index
            int w_l = tid % warpSize;//thread index within a warp
            int t_m = INT_MAX >> (warpSize-w_l-1); //thread mask (ERROR IN THE PAPER minus one is required)

            int b	= __ballot(pred) & t_m; //balres = number whose ith bit isone if the ith's thread pred is true masked up to the current index in warp
            int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

            if(w_l==warpSize-1){
                warpTotals[w_i]=t_u+pred;
            }
            __syncthreads();

            if(w_i==0 && w_l<blockDim.x/warpSize){
                int w_i_u=0;
                for(int j=0;j<=5;j++){
                    int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
                    w_i_u += (__popc(b_j & t_m)  ) << j;
                    //printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
                }
                warpTotals[w_l]=w_i_u;
            }

            __syncthreads();

            if(pred){
                int idx = t_u + warpTotals[w_i] + bo + 4;
                if (idx < limit + 4) {
                    z[idx]= value > static_cast<T>(0.0f) ? tid+1 : -(tid + 1);
                    x[tid] = value > static_cast<T>(0.0f) ? x[tid] - threshold : x[tid] + threshold;
                }
            }
        }
    }

/*
*   This kernel handles decode from sparse threshold array, to dense array
 *
 *   PLEASE NOTE: Z is expected to be memset to 0
*/
    template<typename T>
    __device__ inline void decoderKernelGeneric(void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<int *> (dx);
        auto z = reinterpret_cast<T *> (dz);

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float threshold;
        __shared__ int limit;

        __shared__ FloatBits fb;
        if (threadIdx.x == 0) {
            limit = x[0];
            fb.i_ = x[2];
            threshold = fb.f_;
        }
        __syncthreads();

        for (int e = tid; e < limit; e += blockDim.x * gridDim.x) {
            int el = x[e+4];
            int ael = nd4j::math::nd4j_abs<int>(el) - 1;

            // TODO: investigate, if += would work better here, as in "decoded accumulation"
            z[ael] += el > 0 ? threshold : -threshold;
        }
    }

    template<typename T>
    __device__ inline void cudaDecodeBitmapGeneric(void *dx, Nd4jLong N, T *dz) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ T *shmem;
        __shared__ FloatBits fb;
        __shared__ float threshold;
        __shared__ int *x;
        if (threadIdx.x == 0){
            extern __shared__ char mem[];
            shmem = reinterpret_cast<T*>(mem);
            x = reinterpret_cast<int *>(dx);
            fb.i_ = x[2];
            threshold = fb.f_;
        }
        __syncthreads();

        int lim = N / 16 + 5;
        for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
            int byteId = i / 16 + 4;
//        printf("I: [%i]; byteId: [%i]\n", i, byteId);

            shmem[threadIdx.x] = dz[i];
            __syncthreads();

            if (threadIdx.x % 16 == 0) {
                int byte = x[byteId];

                for (int e = 0; e < 16; e++) {
                    if (i + e >= N)
                        continue;

                    int bitId = (i + e) % 16;

                    bool hasBit = (byte & 1 << (bitId) ) != 0;
                    bool hasSign = (byte & 1 << (bitId + 16) ) != 0;

                    if (hasBit) {
                        if (hasSign)
                            shmem[threadIdx.x + bitId] -= threshold;
                        else
                            shmem[threadIdx.x + bitId] += threshold;
                    } else if (hasSign) {
                        shmem[threadIdx.x + bitId] -= threshold / 2;
                    }
                }
            }
            __syncthreads();

            dz[i] = shmem[threadIdx.x];
        }
    }


    template<typename T>
    __device__ inline void cudaEncodeBitmapGeneric(T *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ int counter;
        __shared__ int *shmem;
        __shared__ T *vals;
        if (threadIdx.x == 0){
            extern __shared__ char mem[];
            shmem = reinterpret_cast<int*>(mem);
            vals = reinterpret_cast<T *>(shmem + blockDim.x);
            counter = 0;
        }
        __syncthreads();

        for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
            // all threads in block reading stuff
            T val = dx[i];
            T abs = nd4j::math::nd4j_abs<T>(val);

            int byteId = i / 16 + 4;
            int bitId = i % 16;

            shmem[threadIdx.x] = 0;
            vals[threadIdx.x] = val;

            if (abs >= static_cast<T>(threshold)) {
                shmem[threadIdx.x] = 1 << (bitId);
                atomicAdd(&counter, 1);
                if (val < static_cast<T>(0.0f)) {
                    shmem[threadIdx.x] |= 1 << (bitId + 16);
                    vals[threadIdx.x] += static_cast<T>(threshold);
                } else {
                    vals[threadIdx.x] -= static_cast<T>(threshold);
                }
            } else if (abs >= static_cast<T>(threshold) / static_cast<T>(2.0f) && val < static_cast<T>(0.0f)) {
                atomicAdd(&counter, 1);
                shmem[threadIdx.x] = 1 << (bitId + 16);

                vals[threadIdx.x] += static_cast<T>(threshold) / static_cast<T>(2.0f);
            }
            __syncthreads();

            if (threadIdx.x % 16 == 0) {
                int byte = 0;
                for (int e = 0; e < 16; e++) {
                    if (i + e >= N)
                        continue;

                    byte |= shmem[threadIdx.x + e];
                }
                dz[byteId] = byte;
            }
            __syncthreads();

            dx[i] = vals[threadIdx.x];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(scalar, counter);
        }
    }

    __global__ void cudaEncodeBitmapFloat(float *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<float>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaEncodeBitmapDouble(double *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<double>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaEncodeBitmapHalf(float16 *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<float16>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaDecodeBitmapFloat(void *dx, Nd4jLong N, float *dz) {
        nd4j::cudaDecodeBitmapGeneric<float>(dx, N, dz);
    }

    __global__ void cudaDecodeBitmapDouble(void *dx, Nd4jLong N, double *dz) {
        nd4j::cudaDecodeBitmapGeneric<double>(dx, N, dz);
    }

    __global__ void cudaDecodeBitmapHalf(void *dx, Nd4jLong N, float16 *dz) {
        nd4j::cudaDecodeBitmapGeneric<float16>(dx, N, dz);
    }


    __global__ void encoderKernelP1Float(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<float>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP1Double(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<double>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP1Half(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<float16>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP2Float(int *dx, Nd4jLong N, int *dz) {
        nd4j::encoderKernelP2Generic<float>(dx, N, dz);
    }

    __global__ void encoderKernelP3Float(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<float>(dx, offsets, N, dz);
    }

    __global__ void encoderKernelP3Double(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<double>(dx, offsets, N, dz);
    }

    __global__ void encoderKernelP3Half(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<float16>(dx, offsets, N, dz);
    }

    __global__ void decoderKernelFloat(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<float>(dx, N, dz);
    }

    __global__ void decoderKernelDouble(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<double>(dx, N, dz);
    }

    __global__ void decoderKernelHalf(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<float16>(dx, N, dz);
    }

    template <typename S, typename T>
    __global__ void convertKernel(void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<S *>(dx);
        auto z = reinterpret_cast<T *>(dz);

        nd4j::convertKernelGeneric(x, N, z);
    }

    template __global__ void prescan<false, false>(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);
    template __global__ void prescan<false, true>(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);
    template __global__ void prescan<true, false>(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);
    template __global__ void prescan<true, true>(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);

    //

    BUILD_DOUBLE_TEMPLATE(template __global__ void convertKernel, (void *dx, Nd4jLong N, void *dz), LIBND4J_TYPES, LIBND4J_TYPES)
}