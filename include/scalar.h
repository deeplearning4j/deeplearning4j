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
	__inline__ __device__ void transform(int n, int idx, T dx, T *dy, int incy, T *params, T *result, int blockSize) {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	ScalarTransform() {
	}
};

namespace ops {
template<typename T>
class Add: public virtual ScalarTransform<T> {
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

#ifdef __CUDACC__
	__host__ __device__
#endif
	Add() {
	}

};

template<typename T>
class Divide: public virtual ScalarTransform<T> {
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

#ifdef __CUDACC__
	__host__ __device__
#endif
	Divide() {
	}

};

template<typename T>
class Equals: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Equals() {
	}

};

template<typename T>
class GreaterThan: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	GreaterThan() {
	}

};

template<typename T>
class LessThan: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	LessThan() {
	}
};

template<typename T>
class LessThanOrEqual: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	LessThanOrEqual() {
	}

};

template<typename T>
class Max: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Max() {
	}
};

template<typename T>
class Min: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Min() {
	}

};

template<typename T>
class Multiply: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Multiply() {
	}

};

template<typename T>
class NotEquals: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	NotEquals() {
	}
};

template<typename T>
class ReverseDivide: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	ReverseDivide() {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Set() {
	}


};

template<typename T>
class Subtract: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	Subtract() {
	}

};

template<typename T>
class SetValOrLess: public virtual ScalarTransform<T> {
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
#ifdef __CUDACC__
	__host__ __device__
#endif
	SetValOrLess() {
	}


};
}

template<typename T>
class ScalarOpFactory {
public:

#ifdef __CUDACC__
	__host__ __device__
#endif
	ScalarOpFactory() {
	}



#ifdef __CUDACC__
	__host__ __device__
#endif
	ScalarTransform<T> * getOp(int op) {
		if (op == 0)
			return new functions::scalar::ops::Add<T>();
		else if (op == 1)
			return new functions::scalar::ops::Subtract<T>();
		else if (op == 2)
			return  new functions::scalar::ops::Multiply<T> ();
		else if (op == 3)
			return new functions::scalar::ops::Divide<T>();
		else if (op == 4)
			return new functions::scalar::ops::ReverseDivide<T>();
		else if (op == 5)
			return new functions::scalar::ops::ReverseSubtract<T>();
		else if (op == 6)
			return new functions::scalar::ops::Max<T> ();
		else if (op == 7)
			return new functions::scalar::ops::LessThan<T> ();
		else if (op == 8)
			return new functions::scalar::ops::GreaterThan<T>();
		else if (op == 9)
			return new functions::scalar::ops::Equals<T>();
		else if (op == 10)
			return new functions::scalar::ops::LessThanOrEqual<T>();
		else if (op == 11)
			return new functions::scalar::ops::NotEquals<T>();
		else if (op == 12)
			return new functions::scalar::ops::Min<T>();
		else if (op == 13)
			return new functions::scalar::ops::Set<T>();
		return NULL;
	}
};

}
}
#ifdef __CUDACC__
__constant__ functions::scalar::ScalarOpFactory<double> *scalarDoubleOpFactory;
__constant__ functions::scalar::ScalarOpFactory<float> *scalarFloatOpFactory;


extern "C"
__host__ void setupScalarTransformFactories() {
/*	printf("Setting up transform factories\n");
	functions::scalar::ScalarOpFactory<double> *newOpFactory =  new functions::scalar::ScalarOpFactory<double>();
	functions::scalar::ScalarOpFactory<float> *newOpFactoryFloat =  new functions::scalar::ScalarOpFactory<float>();
	checkCudaErrors(cudaMemcpyToSymbol(scalarDoubleOpFactory, newOpFactory, sizeof( functions::scalar::ScalarOpFactory<double> )));
	checkCudaErrors(cudaMemcpyToSymbol(scalarFloatOpFactory, newOpFactory, sizeof( functions::scalar::ScalarOpFactory<float>)));
	delete(newOpFactory);
	delete(newOpFactoryFloat);*/

}

template <typename T>
__device__ void scalarGeneric(
		int opNum,
		int n,
		int idx,
		T dx,
		T *dy,
		int incy, T *params,
		T *result, int blockSize) {
	__shared__ functions::scalar::ScalarTransform<T> *op;
	__shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;
	if(threadIdx.x == 0)
		scalarDoubleOpFactory = new functions::scalar::ScalarOpFactory<T>();

	__syncthreads();
	if(threadIdx.x == 0)
		op = scalarDoubleOpFactory->getOp(opNum);
	__syncthreads();




	op->transform(n,idx,dx,dy,incy,params,result,blockSize);
	if(threadIdx.x == 0)
		free(op);
}

extern "C" __global__ void scalarDouble(
		int opNum,
		int n,
		int idx,
		double dx,
		double *dy,
		int incy, double *params,
		double *result, int blockSize) {
	scalarGeneric<double>(opNum,n,idx,dx,dy,incy,params,result,blockSize);
}

extern "C" __global__ void scalarFloat(int opNum,
		int n, int idx, float dx, float *dy, int incy, float *params, float *result, int blockSize) {
	scalarGeneric<float>(opNum,n,idx,dx,dy,incy,params,result,blockSize);
}



#endif
#endif /* SCALAR_H_ */
