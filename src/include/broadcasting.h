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
	__device__ void transform(
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
				if (i < resultLength)
					result[i] = op(x[i], y[yOffset2]);

			}
		} else {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Add() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Copy() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Divide() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Multiply() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	ReverseDivide() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	ReverseSubtract() {
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
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Subtract() {
	}
};
}

template<typename T>
class BroadcastOpFactory {
public:
	BroadcastOpFactory() {
	}
#ifdef __CUDACC__
	__host__
#endif
	Broadcast<T> * getOp(std::string name) {
		return getOp(name.c_str());

	}

#ifdef __CUDACC__
	__host__ __device__
#endif
	Broadcast<T> * getOp(char *name) {
		if (functions::ops::strcmp(name,"add_strided")) {
			return new functions::broadcast::ops::Add<T>();

		} else if (functions::ops::strcmp(name,"sub_strided")) {
			return new functions::broadcast::ops::Subtract<T>();
		} else if (functions::ops::strcmp(name,"mul_strided")) {
			return new  functions::broadcast::ops::Multiply<T>();
		} else if (functions::ops::strcmp(name,"div_strided")) {
			return new functions::broadcast::ops::Divide<T>();
		} else if (functions::ops::strcmp(name,"rdiv_strided")) {
			return new functions::broadcast::ops::ReverseDivide<T>();
		} else if (functions::ops::strcmp(name,"rsub_strided")) {
			return new functions::broadcast::ops::ReverseSubtract<T>();
		} else if (functions::ops::strcmp(name,"copy_strided")) {
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
	functions::broadcast::BroadcastOpFactory<double> *newOpFactory =  functions::broadcast::BroadcastOpFactory<double>();
	functions::broadcast::BroadcastOpFactory<float> *newOpFactoryFloat =  functions::broadcast::BroadcastOpFactory<float>();
	checkCudaErrors(cudaMemcpyToSymbol(broadcastDoubleFactory, newOpFactory, sizeof( functions::broadcast::BroadcastOpFactory<double> )));
	checkCudaErrors(cudaMemcpyToSymbol(broadcastFloatFactory, newOpFactory, sizeof( functions::broadcast::BroadcastOpFactory<float>)));
	delete(newOpFactory);
	delete(newOpFactoryFloat);
}


extern "C" __global__ void broadcastDouble(
		char *name,
		double *x, int *xShapeInfo,
		double *y, int *yShapeInfo,
		double *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	functions::broadcast::Broadcast<double> *op = broadcastDoubleFactory->getOp(name);
	op->transform(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
	free(op);
}

extern "C" __global__ void broadcastFloat(
		char *name,
		float *x, int *xShapeInfo,
		float *y, int *yShapeInfo,
		float *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	functions::broadcast::Broadcast<float> *op = broadcastFloatFactory->getOp(name);
	op->transform(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
	free(op);

}

#endif



#endif /* BROADCASTING_H_ */
