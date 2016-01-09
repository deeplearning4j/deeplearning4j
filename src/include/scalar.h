/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_

#include <op.h>
#include <templatemath.h>
namespace functions {
namespace scalar {
template<typename T>
class ScalarTransform: public virtual functions::ops::Op<T> {

public:
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) = 0;

#ifdef __CUDACC__
	/**
	 *
	 * @param n
	 * @param idx
	 * @param dx
	 * @param dy
	 * @param incy
	 * @param params
	 * @param result
	 * @param blockSize
	 */
	virtual
	__device__ void transform(int n, int idx, T dx, T *dy, int incy, T *params, T *result, int blockSize) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		for (; i < n; i += totalThreads) {
			result[idx + i * incy] = op(dx, dy[idx + i * incy], params);
		}

	}
#endif

	virtual void transform(T *x, int xStride, T *result, int resultStride,
			T scalar, T *extraParams, int n) {
		if (xStride == 1 && resultStride == 1) {
#pragma omp simd

			for (int i = 0; i < n; i++) {
				result[i] = op(x[i], scalar, extraParams);
			}

		} else {
#pragma omp simd
			for (int i = 0; i < n; i++) {
				result[i * resultStride] = op(x[i * resultStride], scalar,
						extraParams);
			}
		}

	}

#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~ScalarTransform() {
	}
};

namespace ops {
template<typename T>
class Add: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 + d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("add_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Add() {
	}

};

template<typename T>
class Divide: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 / d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("div_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Divide() {
	}

};

template<typename T>
class Equals: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 == d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("eq_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Equals() {
	}

};

template<typename T>
class GreaterThan: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 > d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("gt_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~GreaterThan() {
	}

};

template<typename T>
class LessThan: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 < d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("add_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~LessThan() {
	}

};

template<typename T>
class LessThanOrEqual: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 <= d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("ltoreq_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~LessThanOrEqual() {
	}

};

template<typename T>
class Max: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return nd4j::math::nd4j_max<T>(d1, d2);
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("max_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Max() {
	}

};

template<typename T>
class Min: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return nd4j::math::nd4j_min<T>(d1, d2);
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("min_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Min() {
	}

};

template<typename T>
class Multiply: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 * d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() {
		return std::string("mul_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Multiply() {
	}

};

template<typename T>
class NotEquals: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 != d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("noteq_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~NotEquals() {
	}

};

template<typename T>
class ReverseDivide: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d2 / d1;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("rdiv_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~ReverseDivide() {
	}

};

template<typename T>
class ReverseSubtract: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d2 - d1;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("rsib_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~ReverseSubtract() {
	}

};

template<typename T>
class Set: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("set_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Set() {
	}

};

template<typename T>
class Subtract: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		return d1 - d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sub_scalar");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~Subtract() {
	}

};

template<typename T>
class SetValOrLess: public virtual ScalarTransform<T> {
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *params) {
		if (d2 < d1) {
			return d1;
		}
		return d2;
	}
	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("setvalorless_scalar");
	}

#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual inline ~SetValOrLess() {
	}

};
}

template<typename T>
class ScalarOpFactory {
public:
	ScalarOpFactory() {
	}


#ifdef __CUDACC__
	__host__
#endif
	ScalarTransform<T> * getOp(std::string name) {
		return getOp(name.c_str());
	}


#ifdef __CUDACC__
	__host__ __device__
#endif
	ScalarTransform<T> * getOp(char *name) {
		if (functions::ops::strcmp(name,"add_scalar"))
			return (functions::scalar::ops::Add<T> *) malloc(sizeof(functions::scalar::ops::Add<T>));
		else if (functions::ops::strcmp(name,"sub_scalar"))
			return (functions::scalar::ops::Subtract<T> *) malloc(sizeof(functions::scalar::ops::Subtract<T>));
		else if (functions::ops::strcmp(name,"mul_scalar"))
			return  (functions::scalar::ops::Multiply<T> *) malloc(sizeof(functions::scalar::ops::Multiply<T>));
		else if (functions::ops::strcmp(name,"div_scalar"))
			return (functions::scalar::ops::Divide<T> *) malloc(sizeof(functions::scalar::ops::Divide<T>));
		else if (functions::ops::strcmp(name,"rdiv_scalar"))
			return (functions::scalar::ops::ReverseDivide<T> *) malloc(sizeof(functions::scalar::ops::ReverseDivide<T>));
		else if (functions::ops::strcmp(name,"rsub_scalar"))
			return (functions::scalar::ops::ReverseSubtract<T> *) malloc(sizeof(functions::scalar::ops::ReverseSubtract<T>));
		else if (functions::ops::strcmp(name,"max_scalar"))
			return (functions::scalar::ops::Max<T> *) malloc(sizeof(functions::scalar::ops::Max<T>));
		else if (functions::ops::strcmp(name,"lt_scalar"))
			return (functions::scalar::ops::LessThan<T> *) malloc(sizeof(functions::scalar::ops::LessThan<T>));
		else if (functions::ops::strcmp(name,"gt_scalar"))
			return (functions::scalar::ops::GreaterThan<T> *) malloc(sizeof(functions::scalar::ops::GreaterThan<T>));
		else if (functions::ops::strcmp(name,"eq_scalar"))
			return (functions::scalar::ops::Equals<T> *) malloc(sizeof(functions::scalar::ops::Equals<T>));
		else if (functions::ops::strcmp(name,"lessthanorequal_scalar"))
			return ( functions::scalar::ops::LessThanOrEqual<T>* ) malloc(sizeof(functions::scalar::ops::LessThanOrEqual<T>));
		else if (functions::ops::strcmp(name,"neq_scalar"))
			return (functions::scalar::ops::NotEquals<T> *) malloc(sizeof(functions::scalar::ops::NotEquals<T>));
		else if (functions::ops::strcmp(name,"min_scalar"))
			return (functions::scalar::ops::Min<T> *) malloc(sizeof(functions::scalar::ops::Min<T>));
		else if (functions::ops::strcmp(name,"set_scalar"))
			return (functions::scalar::ops::Set<T> *) malloc(sizeof(functions::scalar::ops::Set<T>));
		return NULL;
	}
};

}
}
#ifdef __CUDACC__
__constant__ functions::scalar::ScalarOpFactory<double> *scalarDoubleOpFactory;
__constant__ functions::scalar::ScalarOpFactory<float> *scalarFloatOpFactory;

extern "C" __global__ void scalarDouble(
		char *name,
		int n,
		int idx,
		double dx,
		double *dy,
		int incy, double *params,
		double *result, int blockSize) {
	functions::scalar::ScalarTransform<double> *op = scalarDoubleOpFactory->getOp(name);
	op->transform(n,idx,dx,dy,incy,params,result,blockSize);
}

extern "C" __global__ void scalarFloat(char *name,
		int n, int idx, float dx, float *dy, int incy, float *params, float *result, int blockSize) {
	functions::scalar::ScalarTransform<float> *op = scalarFloatOpFactory->getOp(name);
	op->transform(n,idx,dx,dy,incy,params,result,blockSize);

}

#endif
#endif /* SCALAR_H_ */
