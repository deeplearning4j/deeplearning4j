//
// Created by agibsonccc on 2/21/16.
//

#ifndef NATIVEOPERATIONS_CUDA_LAUNCH_CONFIG_H
#define NATIVEOPERATIONS_CUDA_LAUNCH_CONFIG_H
/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <helper_string.h>
#include <helper_cuda.h>
#include <dll.h>

/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator".
 *  \note The __global__ function of interest is presumed to use 0 bytes of dynamically-allocated __shared__ memory.
 */
inline
#ifdef __CUDACC__
__host__ __device__
#endif
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties);

/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  Use this version of the function when a CUDA block's dynamically-allocated __shared__ memory requirements
 *  vary with the size of the block.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \param block_size_to_dynamic_smem_bytes A unary function which maps an integer CUDA block size to the number of bytes
 *         of dynamically-allocated __shared__ memory required by a CUDA block of that size.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator".
 */
template<typename UnaryFunction>
#ifdef __CUDACC__
__host__ __device__
#endif
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size);



namespace __cuda_launch_config_detail
{

    using std::size_t;

    namespace util
    {


        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
        T min_(const T &lhs, const T &rhs)
        {
            return rhs < lhs ? rhs : lhs;
        }


        template <typename T>
        struct zero_function
        {
            inline __host__ __device__
            T operator()(T)
            {
                return 0;
            }
        };


// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
        template<typename L, typename R>
#ifdef __CUDACC__
        __host__ __device__
#endif
        L divide_ri(const L x, const R y)
        {
            return (x + (y - 1)) / y;
        }

// x/y rounding towards zero for integers, used to determine # of blocks/warps etc.
        template<typename L, typename R>
#ifdef __CUDACC__
        __host__ __device__
#endif
        L divide_rz(const L x, const R y)
        {
            return x / y;
        }

// round x towards infinity to the next multiple of y
        template<typename L, typename R>
        inline
#ifdef __CUDACC__
        __host__ __device__
#endif
        L round_i(const L x, const R y){ return y * divide_ri(x, y); }

// round x towards zero to the next multiple of y
        template<typename L, typename R>
        inline
#ifdef __CUDACC__
        __host__ __device__
#endif
        L round_z(const L x, const R y){ return y * divide_rz(x, y); }

    } // end namespace util



// granularity of shared memory allocation
    inline
#ifdef __CUDACC__
    __host__ __device__
#endif

    size_t smem_allocation_unit(const cudaDeviceProp &properties)
    {
        switch(properties.major)
        {
            case 1:  return 512;
            case 2:  return 128;
            case 3:  return 256;
            default: return 256; // unknown GPU; have to guess
        }
    }


// granularity of register allocation
    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t reg_allocation_unit(const cudaDeviceProp &properties, const size_t regsPerThread)
    {
        switch(properties.major)
        {
            case 1:  return (properties.minor <= 1) ? 256 : 512;
            case 2:  switch(regsPerThread)
                {
                    case 21:
                    case 22:
                    case 29:
                    case 30:
                    case 37:
                    case 38:
                    case 45:
                    case 46:
                        return 128;
                    default:
                        return 64;
                }
            case 3:  return 256;
            default: return 256; // unknown GPU; have to guess
        }
    }


// granularity of warp allocation
    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t warp_allocation_multiple(const cudaDeviceProp &properties)
    {
        return (properties.major <= 1) ? 2 : 1;
    }

// number of "sides" into which the multiprocessor is partitioned
    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t num_sides_per_multiprocessor(const cudaDeviceProp &properties)
    {
        switch(properties.major)
        {
            case 1:  return 1;
            case 2:  return 2;
            case 3:  return 4;
            default: return 4; // unknown GPU; have to guess
        }
    }


    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t max_blocks_per_multiprocessor(const cudaDeviceProp &properties)
    {
        return (properties.major <= 2) ? 8 : 16;
    }


    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp     &properties,
                                                const cudaFuncAttributes &attributes,
                                                size_t CTA_SIZE,
                                                size_t dynamic_smem_bytes)
    {
        // Determine the maximum number of CTAs that can be run simultaneously per SM
        // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet

        //////////////////////////////////////////
        // Limits due to threads/SM or blocks/SM
        //////////////////////////////////////////
        const size_t maxThreadsPerSM = properties.maxThreadsPerMultiProcessor;  // 768, 1024, 1536, etc.
        const size_t maxBlocksPerSM  = max_blocks_per_multiprocessor(properties);

        // Calc limits
        const size_t ctaLimitThreads = (CTA_SIZE <= properties.maxThreadsPerBlock) ? maxThreadsPerSM / CTA_SIZE : 0;
        const size_t ctaLimitBlocks  = maxBlocksPerSM;

        //////////////////////////////////////////
        // Limits due to shared memory/SM
        //////////////////////////////////////////
        const size_t smemAllocationUnit     = smem_allocation_unit(properties);
        const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
        const size_t smemPerCTA = util::round_i(smemBytes, smemAllocationUnit);

        // Calc limit
        const size_t ctaLimitSMem = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;

        //////////////////////////////////////////
        // Limits due to registers/SM
        //////////////////////////////////////////
        const size_t regAllocationUnit      = reg_allocation_unit(properties, attributes.numRegs);
        const size_t warpAllocationMultiple = warp_allocation_multiple(properties);
        const size_t numWarps = util::round_i(util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

        // Calc limit
        size_t ctaLimitRegs;
        if(properties.major <= 1)
        {
            // GPUs of compute capability 1.x allocate registers to CTAs
            // Number of regs per block is regs per thread times number of warps times warp size, rounded up to allocation unit
            const size_t regsPerCTA = util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);
            ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA : maxBlocksPerSM;
        }
        else
        {
            // GPUs of compute capability 2.x and higher allocate registers to warps
            // Number of regs per warp is regs per thread times times warp size, rounded up to allocation unit
            const size_t regsPerWarp = util::round_i(attributes.numRegs * properties.warpSize, regAllocationUnit);
            const size_t numSides = num_sides_per_multiprocessor(properties);
            const size_t numRegsPerSide = properties.regsPerBlock / numSides;
            ctaLimitRegs = regsPerWarp > 0 ? ((numRegsPerSide / regsPerWarp) * numSides) / numWarps : maxBlocksPerSM;
        }

        //////////////////////////////////////////
        // Overall limit is min() of limits due to above reasons
        //////////////////////////////////////////
        return util::min_(ctaLimitRegs, util::min_(ctaLimitSMem, util::min_(ctaLimitThreads, ctaLimitBlocks)));
    }


    template <typename UnaryFunction>
    inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    size_t default_block_size(const cudaDeviceProp     &properties,
                              const cudaFuncAttributes &attributes,
                              UnaryFunction block_size_to_smem_size)
    {
        size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
        size_t largest_blocksize  = util::min_(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
        size_t granularity        = properties.warpSize;
        size_t max_blocksize      = 0;
        size_t highest_occupancy  = 0;

        for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
        {
            size_t occupancy = blocksize * max_active_blocks_per_multiprocessor(properties, attributes, blocksize, block_size_to_smem_size(blocksize));

            if(occupancy > highest_occupancy)
            {
                max_blocksize = blocksize;
                highest_occupancy = occupancy;
            }

            // early out, can't do better
            if(highest_occupancy == max_occupancy)
                break;
        }

        return max_blocksize;
    }


} // end namespace __cuda_launch_config_detail


template<typename UnaryFunction>
inline
#ifdef __CUDACC__
__host__ __device__
#endif

std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size)
{
    return __cuda_launch_config_detail::default_block_size(properties, attributes, block_size_to_dynamic_smem_size);
}


inline
#ifdef __CUDACC__
__host__ __device__
#endif
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties)
{
    return block_size_with_maximum_potential_occupancy(attributes, properties, __cuda_launch_config_detail::util::zero_function<std::size_t>());
}

template<typename T>
inline
#ifdef __CUDACC__
__host__
#endif
std::size_t block_size_with_maximum_potential_occupancy(T t)
{
    cudaFuncAttributes attributes;
    checkCudaErrors(cudaFuncGetAttributes(&attributes, t));
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, device));
    return block_size_with_maximum_potential_occupancy(attributes, properties);
}
#endif //NATIVEOPERATIONS_CUDA_LAUNCH_CONFIG_H
