/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_

#include <op.h>
namespace functions {
namespace pairwise_transforms {
#define MIN 1e-12

template<typename T>
class PairWiseTransform: public virtual functions::ops::Op<T> {
public:
	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) = 0;

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) = 0;

#ifdef __CUDACC__
	/**
	 *
	 * @param n
	 * @param xOffset
	 * @param yOffset
	 * @param resultOffset
	 * @param dx
	 * @param dy
	 * @param incx
	 * @param incy
	 * @param params
	 * @param result
	 * @param incz
	 * @param blockSize
	 */
	virtual __device__ void transform(
			int n,
			int xOffset,
			int yOffset,
			int resultOffset,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result, int incz, int blockSize) {

		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		if (incy == 0) {
			if ((blockIdx.x == 0) && (tid == 0)) {
				for (; i < n; i++) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], params);
				}

			}
		} else if ((incx == incy) && (incx > 0)) {
			/* equal, positive, increments */
			if (incx == 1) {
				/* both increments equal to 1 */
				for (; i < n; i += totalThreads) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
							params);
				}
			} else {
				/* equal, positive, non-unit increments. */
				for (; i < n; i += totalThreads) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
							params);
				}
			}
		} else {
			/* unequal or nonpositive increments */
			for (; i < n; i += totalThreads) {
				result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
						params);
			}
		}
	}

#endif
public:
	virtual void exec(T *dx, int xStride, T *y, int yStride, T *result,
			int resultStride, T *extraParams, int n) {
		if (xStride == 1 && yStride == 1 && resultStride == 1) {
#pragma omp simd
			for (int i = 0; i < n; i++) {
				result[i] = op(dx[i], y[i], extraParams);
			}

		} else {
#pragma omp simd
			for (int i = 0; i < n; i++) {
				result[i * resultStride] = op(dx[i * resultStride],
						y[i * yStride], extraParams);
			}
		}
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~PairWiseTransform() {
	}

};

namespace ops {
template<typename T>
class Add: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("add_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 + d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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
class Copy: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("copy_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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
class Divide: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() {

		return std::string("div_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 / d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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
class Epsilon: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("eps_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		T diff = d1 - d2;
		T absDiff = abs(diff);
		if (absDiff < MIN)
			return 1;
		return 0;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~Epsilon() {
	}
};

template<typename T>
class EqualTo: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("eq_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 != d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~EqualTo() {
	}
};

template<typename T>
class GreaterThan: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("gt_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 > d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~GreaterThan() {
	}
};

template<typename T>
class LessThan: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("lt_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 < d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~LessThan() {
	}
};

template<typename T>
class Multiply: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("mul_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 * d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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
class ReverseDivide: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("rdiv_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d2 / d1;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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
class ReverseSubtraction: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("rsub_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d2 - d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~ReverseSubtraction() {
	}
};

template<typename T>
class Subtract: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sub_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 - d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
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

template<typename T>
class Softmax: public virtual PairWiseTransform<T> {

	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() {
		return std::string("softmax_strided");
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T d2, T *params) {
		return d1 / d2;
	}

	virtual
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	T op(T d1, T *params) {
		return d1;
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline

#endif
	virtual ~Softmax() {
	}
};
}

template<typename T>
class PairWiseTransformOpFactory {
public:
	PairWiseTransformOpFactory() {
	}

#ifdef __CUDACC__
	__host__ __device__
#endif
	PairWiseTransform<T> *getOp(char *name) {
		if (functions::ops::strcmp(name,"add_strided"))
			return (pairwise_transforms::ops::Add<T> *) malloc(sizeof(pairwise_transforms::ops::Add<T>));
		if (functions::ops::strcmp(name,"copy_strided"))
			return (pairwise_transforms::ops::Copy<T>*) malloc(sizeof(pairwise_transforms::ops::Copy<T>));
		if (functions::ops::strcmp(name,"div_strided"))
			return (pairwise_transforms::ops::Divide<T> *) malloc(sizeof(pairwise_transforms::ops::Divide<T>));
		if (functions::ops::strcmp(name,"eps_strided"))
			return (pairwise_transforms::ops::Divide<T>* )malloc(sizeof(pairwise_transforms::ops::Epsilon<T>));
		if (functions::ops::strcmp(name, "eq_strided"))
			return (pairwise_transforms::ops::EqualTo<T> *) malloc(sizeof(pairwise_transforms::ops::EqualTo<T>));
		if (functions::ops::strcmp(name,"gt_strided"))
			return ( pairwise_transforms::ops::GreaterThan<T> *) malloc(sizeof(pairwise_transforms::ops::GreaterThan<T>));
		if (functions::ops::strcmp(name,"lt_strided"))
			return (pairwise_transforms::ops::LessThan<T> *) malloc(sizeof(pairwise_transforms::ops::LessThan<T>));
		if (functions::ops::strcmp(name,"mul_strided"))
			return ( pairwise_transforms::ops::Multiply<T> *) malloc(sizeof(pairwise_transforms::ops::Multiply<T>));
		if (functions::ops::strcmp(name,"div_strided"))
			return (pairwise_transforms::ops::Divide<T> *) malloc(sizeof(pairwise_transforms::ops::Divide<T>));
		if (functions::ops::strcmp(name,"rdiv_strided"))
			return (pairwise_transforms::ops::ReverseDivide<T> *) malloc(sizeof(new pairwise_transforms::ops::ReverseDivide<T>));
		if (functions::ops::strcmp(name,"rsub_strided"))
			return (pairwise_transforms::ops::ReverseSubtraction<T> *) malloc(sizeof(pairwise_transforms::ops::ReverseSubtraction<T>));
		if (functions::ops::strcmp(name,"sub_strided"))
			return (pairwise_transforms::ops::Subtract<T> *) malloc(sizeof(pairwise_transforms::ops::Subtract<T>));
		return NULL;
	}

#ifdef __CUDACC__
	__host__
#endif
	PairWiseTransform<T> *getOp(std::string name) {
		return getOp(name.c_str());
	}

};
}
}

#ifdef __CUDACC__
__constant__ functions::pairwise_transforms::PairWiseTransformOpFactory<double> *pairWiseDoubleFactory;
__constant__ functions::pairwise_transforms::PairWiseTransformOpFactory<float> *pairWiseFloatFactory;

extern "C" __global__ void pairWiseTransformDouble(
		char *name,
		int n,
		int xOffset,
		int yOffset,
		int resultOffset,
		double *dx,
		double *dy,
		int incx,
		int incy,
		double *params,
		double *result, int incz, int blockSize) {
	functions::pairwise_transforms::PairWiseTransform<double> *op = pairWiseDoubleFactory->getOp(name);
	op->transform(n,xOffset,yOffset,resultOffset,dx,dy,incx,incy,params,result,incz,blockSize);
	free(op);

}



extern "C" __global__ void pairWiseTransformFloat(
		char *name,
		int n,
		int xOffset,
		int yOffset,
		int resultOffset,
		float *dx,
		float *dy,
		int incx,
		int incy,
		float *params,
		float *result, int incz, int blockSize) {
	functions::pairwise_transforms::PairWiseTransform<float> *op = pairWiseFloatFactory->getOp(name);
	op->transform(n,xOffset,yOffset,resultOffset,dx,dy,incx,incy,params,result,incz,blockSize);
	free(op);

}

#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
