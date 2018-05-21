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
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
/*
template<typename T>
struct SharedMemory {
	// Ensure that we won't compile any un-specialized types
	__device__ T
	*

	getPointer() {
		extern __device__ void error(void);
		error();
		return nullptr;
	}

	// Ensure that we won't compile any un-specialized types
	__device__ T
	*

	getPointer(int num) {
		extern __device__ void error(void);
		error();
		return nullptr;
	}
};
*/
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.
/*
template<>
struct SharedMemory<int> {
	__device__ int *getPointer() {
		extern __shared__ int s_int[];
		return s_int;
	}

	__device__ int *getPointer(int val) {
		__shared__ int s_int[10];
		return s_int;
	}
};

template<>
struct SharedMemory<unsigned int> {
	__device__ unsigned int *getPointer() {
		extern __shared__ unsigned int s_uint[];
		return s_uint;
	}
};

template<>
struct SharedMemory<char> {
	__device__ char *getPointer() {
		extern __shared__ char s_char[];
		return s_char;
	}
};

template<>
struct SharedMemory<unsigned char> {
	__device__ unsigned char *getPointer() {
		extern __shared__ unsigned char s_uchar[];
		return s_uchar;
	}
};

template<>
struct SharedMemory<short> {
	__device__ short *getPointer() {
		extern __shared__ short s_short[];
		return s_short;
	}
};

template<>
struct SharedMemory<unsigned short> {
	__device__ unsigned short *getPointer() {
		extern __shared__ unsigned short s_ushort[];
		return s_ushort;
	}
};

template<>
struct SharedMemory<long> {
	__device__ long *getPointer() {
		extern __shared__ long s_long[];
		return s_long;
	}
};

template<>
struct SharedMemory<unsigned long> {
	__device__ unsigned long *getPointer() {
		extern __shared__ unsigned long s_ulong[];
		return s_ulong;
	}
};

template<>
struct SharedMemory<bool> {
	__device__ bool *getPointer() {
		extern __shared__ bool s_bool[];
		return s_bool;
	}
};

template<>
struct SharedMemory<float> {
	__device__ float *getPointer() {
		extern __shared__ float s_float[];
		return s_float;
	}

	__device__ float *getPointer(int val) {
		__shared__ float s_float55[10];
		return s_float55;
	}
};

template<>
struct SharedMemory<double> {
	__device__ double *getPointer() {
		extern __shared__ double s_double[];
		return s_double;
	}

	__device__ double *getPointer(int val) {
		__shared__ double s_double55[10];
		return s_double55;
	}
};
*/
#endif //_SHAREDMEM_H_
#endif