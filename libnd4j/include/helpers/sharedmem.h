/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation and
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 *
 */
#ifdef __CUDACC__
#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <dll.h>

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//
//   With this
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      SharedMemory<T> smem;
//      T* sdata = smem.getPointer();
//      ...
//      doStuff(sdata);
//      ...
//   }
//****************************************************************************

class UnifiedSharedMemory{
    // we accept whole buffer at once
    protected:
    int *sharedMemory;

    int *tBuffer1;
    int *tBuffer2;
    int *tBuffer3;
    int *tBuffer4;
    int *tShared;

    short unifiedSize;
    short factorySize;
    short functionSize;
    short tadSize;


    public:
    __device__ UnifiedSharedMemory(int *shMem) {
        sharedMemory = shMem;
    }

    __device__ __host__ inline void init (int unifiedSize, int factorySize, int functionSize, int tadSize, int xRank) {
        this->unifiedSize = unifiedSize < 16 ? 16 : unifiedSize;
        this->factorySize = factorySize < 16 ? 16 : factorySize;
        this->functionSize = functionSize < 16 ? 16 : functionSize;
        this->tadSize = tadSize < 16 ? 16 : tadSize;

        this->tBuffer1 =  sharedMemory + ((this->unifiedSize + this->factorySize + this->functionSize + tadSize) / 4);
        this->tBuffer2 = this->tBuffer1 + xRank;
        this->tBuffer3 = this->tBuffer2 + xRank;
        this->tBuffer4 = this->tBuffer3 + xRank;
        this->tShared =  (this->tBuffer4 + xRank);

        //printf("Creating USM<T> -> seflSize: [%i], factorySize: [%i], functionSize: [%i], tadSize: [%i], totalOffset: [%i]\n", this->unifiedSize, this->factorySize, this->functionSize, this->tadSize, this->allocationOffset);
    }

    __device__ __host__ ~UnifiedSharedMemory() { }

    __device__ __host__ inline unsigned char * getUnifiedSpace() {
       return (unsigned char * ) ((int *)sharedMemory);
    }

    __device__ __host__ inline unsigned char * getFactorySpace() {
       return (unsigned char * ) ((int *) getUnifiedSpace() + (unifiedSize / 4));
    }

    __device__ __host__ inline unsigned char * getFunctionSpace() {
       return (unsigned char * ) ((int *)getFactorySpace()  + (factorySize  / 4));
    }

    __device__ __host__ inline unsigned char * getTADSpace() {
       return (unsigned char * ) ((int *)getFunctionSpace() + (functionSize / 4));
    }

    __device__ __host__ inline int * getTempRankBuffer1() {
        return this->tBuffer1;
    }

    __device__ __host__ inline int * getTempRankBuffer2() {
        return this->tBuffer2;
    }

    __device__ __host__ inline int * getTempRankBuffer3() {
        return this->tBuffer3;
    }

    __device__ __host__ inline int * getTempRankBuffer4() {
        return this->tBuffer4;
    }

    __device__ __host__ inline int * getSharedReductionBuffer() {
        return this->tShared;
    }
};

#endif //_SHAREDMEM_H_
#endif