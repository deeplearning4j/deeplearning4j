/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
#include <stdio.h>
#include <sharedmem.h>
#include <shape.h>
#include <op.h>
#include <templatemath.h>
#include <helper_cuda.h>
namespace functions {
namespace broadcast {

template<typename T>
class Broadcast: public functions::ops::Op<T> {
public:

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __device__  __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) = 0;
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __device__  __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) = 0;

#ifdef __CUDACC__
	__inline__ __device__ void transform(
			T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result, int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int *gpuInformation) {

		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		int xOffset = shape::offset(xShapeInfo);
		int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
		int yOffset = shape::offset(yShapeInfo);

		//length for the tad
		int yLength = shape::length(yShapeInfo);
		//length for the tad
		int xLength = shape::length(xShapeInfo);

		int resultLength = shape::length(resultShapeInfo);
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
				i < resultLength;
				i += blockDim.x * gridDim.x) {
			int yOffset2 = yOffset + ((i / xElementWiseStride) % yLength) * yElementWiseStride;
			if (i < resultLength)
				result[i] = op(x[i], y[yOffset2]);

		}

	}
#endif

	virtual void exec(T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result,
			int *resultShapeInfo, int *dimension, int dimensionLength) {

		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
		int yOffset = shape::offset(yShapeInfo);

		//length for the tad
		int yLength = shape::length(yShapeInfo);
		//length for the tad
		int xLength = shape::length(xShapeInfo);

		int resultLength = shape::length(resultShapeInfo);
		if (xElementWiseStride == 1 && yElementWiseStride == 1) {
#pragma omp simd
			for (int i = 0; i < xLength; i++) {
				int yOffset2 = yOffset
						+ ((i / xElementWiseStride) % yLength)
						* yElementWiseStride;
				if (i < resultLength) {
					result[i] = op(x[i], y[yOffset2]);
				}

			}
		}

		else {
#pragma omp simd
			for (int i = 0; i < xLength; i++) {
				int yOffset2 = yOffset
						+ ((i / xElementWiseStride) % yLength)
						* yElementWiseStride;
				if (i < resultLength)
					result[i] = op(x[i], y[yOffset2]);

			}
		}

	}

#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Broadcast() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Broadcast() {
	}

};

namespace ops {
template<typename T>
class Add: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() override {
		return std::string("add");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d1 + d2;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Add() {
	}
};

template<typename T>
class Copy: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() override {
		return std::string("copy");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d2;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Copy() {
	}

};

template<typename T>
class Divide: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() override {
		return std::string("div");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d1 / d2;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Divide() {
	}

};

template<typename T>
class Multiply: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() override {
		return std::string("mul");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d1 * d2;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Multiply() {
	}

};

template<typename T>
class ReverseDivide: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() override {
		return std::string("rdiv");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d2 / d1;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~ReverseDivide() {
	}

};

template<typename T>
class ReverseSubtract: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() override {
		return std::string("rsub");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d2 - d1;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~ReverseSubtract() {
	}

};

template<typename T>
class Subtract: public virtual functions::broadcast::Broadcast<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() override {
		return std::string("sub");
	}

	/**
	 *
	 * @param d1
	 * @param d2
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T d2) {
		return d1 - d2;
	}
	/**
	 *
	 * @param d1
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Subtract() {
	}

};
}

template<typename T>
class BroadcastOpFactory {
public:

#ifdef __CUDACC__
	__host__ __device__
#endif
	BroadcastOpFactory() {
	}


#ifdef __CUDACC__
	__host__ __device__
#endif
	Broadcast<T> * getOp(int op) {
		if (op == 0) {
			return new functions::broadcast::ops::Add<T>();

		} else if (op == 1) {
			return new functions::broadcast::ops::Subtract<T>();
		} else if (op == 2) {
			return new  functions::broadcast::ops::Multiply<T>();
		} else if (op == 3) {
			return new functions::broadcast::ops::Divide<T>();
		} else if (op == 4) {
			return new functions::broadcast::ops::ReverseDivide<T>();
		} else if (op == 5) {
			return new functions::broadcast::ops::ReverseSubtract<T>();
		} else if (op == 6) {
			return new functions::broadcast::ops::Copy<T>();
		}

		return NULL;

	}

};

}
}

#ifdef __CUDACC__

__constant__ functions::broadcast::BroadcastOpFactory<double> *broadcastDoubleFactory;
__constant__ functions::broadcast::BroadcastOpFactory<float> *broadcastFloatFactory;


extern "C"
__host__ void setupBroadcastFactories() {
	printf("Setting up transform factories\n");
/*	functions::broadcast::BroadcastOpFactory<double> *newOpFactory =  new functions::broadcast::BroadcastOpFactory<double>();
	functions::broadcast::BroadcastOpFactory<float> *newOpFactoryFloat =  new functions::broadcast::BroadcastOpFactory<float>();
	checkCudaErrors(cudaMemcpyToSymbol(broadcastDoubleFactory, newOpFactory, sizeof( functions::broadcast::BroadcastOpFactory<double> )));
	checkCudaErrors(cudaMemcpyToSymbol(broadcastFloatFactory, newOpFactory, sizeof( functions::broadcast::BroadcastOpFactory<float>)));
	delete(newOpFactory);
	delete(newOpFactoryFloat);*/
}

template <typename T>
__device__ void broadcastGeneric(
		int opNum,
		T *x, int *xShapeInfo,
		T *y, int *yShapeInfo,
		T *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	__shared__ functions::broadcast::Broadcast<T> *op;
	__shared__ functions::broadcast::BroadcastOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory =  new functions::broadcast::BroadcastOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0) {
		op = newOpFactory->getOp(opNum);
	}
	__syncthreads();


	op->transform(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
	if(threadIdx.x == 0) {
		free(op);
		free(newOpFactory);
	}
}
extern "C" __global__ void broadcastDouble(
		int opNum,
		double *x, int *xShapeInfo,
		double *y, int *yShapeInfo,
		double *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	broadcastGeneric<double>(opNum,x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);

}

extern "C" __global__ void broadcastFloat(
		int opNum,
		float *x, int *xShapeInfo,
		float *y, int *yShapeInfo,
		float *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	broadcastGeneric<float>(opNum,x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);

}

#endif



#endif /* BROADCASTING_H_ */
