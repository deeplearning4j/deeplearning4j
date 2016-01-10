/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <templatemath.h>
#include <op.h>
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif
namespace functions {
namespace transform {

template<typename T>
class Transform: public virtual functions::ops::Op<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) = 0;

#ifdef __CUDACC__
	__device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;
		/* equal, positive, non-unit increments. */
		for (; i < n; i += totalThreads) {
			result[i * incy] = op(dy[i * incy], params);
		}

	}
#endif

public:
	virtual void exec(T *dx, int xStride, T *result, int resultStride,
			T *extraParams, int n) {
		if (xStride == 1 && resultStride == 1) {
#pragma omp simd
			for (int i = 0; i < n; i++) {
				result[i] = op(dx[i], extraParams);
			}

		} else {
#pragma omp simd
			for (int i = 0; i < n; i++) {
				result[i * resultStride] = op(dx[i * resultStride],
						extraParams);
			}
		}

	}

#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Transform() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Transform() {
	}

};

namespace ops {
template<typename T>
class Abs: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_abs<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() {
		return std::string("abs_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Abs() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Abs() {
	}
};

template<typename T>
class Ceiling: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_ceil<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("ceil_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Ceiling() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Ceiling() {
	}
};

template<typename T>
class Cosine: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_cos<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */

#ifdef __CUDACC__
	inline __host__
	virtual
#elif defined(__GNUC__)
	__always_inline
#endif
	std::string name() {
		return std::string("cos_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Cosine() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Cosine() {
	}
};

template<typename T>
class Exp: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_exp<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	std::string name() {
		return std::string("exp_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Exp() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Exp() {
	}
};

template<typename T>
class Floor: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_floor<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("floor_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Floor() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Floor() {
	}
};

template<typename T>
class Log: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_log<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	std::string name() {
		return std::string("log_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Log() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Log() {
	}
};

template<typename T>
class Neg: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return -d1;
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("neg_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Neg() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Neg() {
	}
};

template<typename T>
class Pow: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_pow<T>(d1, params[0]);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("pow_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Pow() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Pow() {
	}
};

template<typename T>
class Round: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_round<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("round_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Round() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Round() {
	}
};

template<typename T>
class Sigmoid: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_sigmoid<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sigmoid_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Sigmoid() {
	}

#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Sigmoid() {
	}
};

template<typename T>
class SetRange: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		T min = params[0];
		T max = params[1];
		if (d1 >= min && d1 <= max)
			return d1;
		if (min == 0 && max == 1) {
			T val = 1 / (1 + nd4j::math::nd4j_exp<T>(-d1));
			return (nd4j::math::nd4j_floor<T>(val * (max - min)) + min);
		}

		T ret = (nd4j::math::nd4j_floor<T>(d1 * (max - min)) + min);
		return ret;
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	std::string name() {
		return std::string("setrange_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~SetRange() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	SetRange() {
	}
};

template<typename T>
class Sin: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_sin<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sin_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Sin() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Sin() {
	}
};

template<typename T>
class Sqrt: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_sqrt<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sqrt_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Sqrt() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Sqrt() {
	}
};

template<typename T>
class SoftPlus: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::softplus<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("softplus_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~SoftPlus() {
	}

#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	SoftPlus() {
	}
};

template<typename T>
class Sign: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return (d1 > 0) - (d1 < 0);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("sign_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Sign() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Sign() {
	}
};

template<typename T>
class Tanh: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_tanh<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("tanh_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~Tanh() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Tanh() {
	}
};

template<typename T>
class ACos: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_acos<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	std::string name() {
		return std::string("acos_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~ACos() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	ACos() {
	}
};

template<typename T>
class ASin: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_asin<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("asin_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~ASin() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	ASin() {
	}
};

template<typename T>
class ATan: public virtual Transform<T> {
public:
	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) {
		return nd4j::math::nd4j_atan<T>(d1);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("atan_strided");
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~ATan() {
	}
#ifdef __CUDACC__
	inline __host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	ATan() {
	}
};

}

template<typename T>
class TransformOpFactory {
public:
#ifdef __CUDACC__
	__device__ __host__
#endif
	TransformOpFactory() {
	}


#ifdef __CUDACC__
	__device__ __host__
#endif
	Transform<T> * getOp(char *name) {
		if (functions::ops::strcmp(name,"abs_strided") == 0) {
			return new transform::ops::Abs<T>();
		}
		if (functions::ops::strcmp(name,"ceil_strided") == 0) {
			return new transform::ops::Ceiling<T> ();
		}
		if (functions::ops::strcmp(name,"cos_strided") == 0) {
			return new transform::ops::Cosine<T>();
		}
		if (functions::ops::strcmp(name,"exp_strided") == 0) {
			return new transform::ops::Exp<T>();
		}
		if (functions::ops::strcmp(name,"floor_strided") == 0) {
			return new transform::ops::Floor<T>();
		}
		if (functions::ops::strcmp(name,"log_strided") == 0) {
			return new transform::ops::Log<T>();
		}
		if (functions::ops::strcmp(name,"neg_strided") == 0) {
			return new transform::ops::Neg<T>();
		}
		if (functions::ops::strcmp(name,"pow_strided") == 0) {
			return new transform::ops::Pow<T>();
		}
		if (functions::ops::strcmp(name,"round_strided") == 0) {
			return  new transform::ops::Round<T>();
		}
		if (functions::ops::strcmp(name,"setrange_strided") == 0) {
			return new transform::ops::SetRange<T>();
		}
		if (functions::ops::strcmp(name,"sigmoid_strided") == 0) {
			return  new transform::ops::Sigmoid<T>();
		}
		if (functions::ops::strcmp(name,"sign_strided") == 0) {
			return new transform::ops::Sign<T>();
		}
		if (functions::ops::strcmp(name,"sin_strided") == 0) {
			return new transform::ops::Sin<T>();
		}
		if (functions::ops::strcmp(name,"softplus_strided") == 0) {
			return new transform::ops::SoftPlus<T>();
		}
		if (functions::ops::strcmp(name,"sqrt_strided") == 0) {
			return new transform::ops::Sqrt<T>();
		}
		if (functions::ops::strcmp(name,"tanh_strided") == 0) {
			return new transform::ops::Tanh<T>();
		}
		if (functions::ops::strcmp(name,"acos_strided") == 0) {
			return new transform::ops::ACos<T> ();
		}
		if (functions::ops::strcmp(name,"asin_strided") == 0) {
			return new transform::ops::ASin<T>();
		}
		if (functions::ops::strcmp(name,"atan_strided") == 0) {
			return new transform::ops::ATan<T>();
		}
		return NULL;
	}



};

}

}

#ifdef __CUDACC__
__device__ __constant__ functions::transform::TransformOpFactory<double> *doubleTransformFactory;
 __device__ __constant__ functions::transform::TransformOpFactory<float> *floatTransformFactory;


extern "C" __global__ void transformDouble(
		char *name,
		int n,
		int idx,
		double *dy,
		int incy,
		double *params,
		double *result, int blockSize) {

	functions::transform::Transform<double> *op = doubleTransformFactory->getOp(name);
	op->transform(n,idx,dy,incy,params,result,blockSize);
	free(op);
}

extern "C" __global__ void transformFloat(
		char *name,
		int n,
		int idx,
		float *dy,
		int incy,
		float *params,
		float *result, int blockSize) {

	functions::transform::Transform<float> *op = floatTransformFactory->getOp(name);
	op->transform(n,idx,dy,incy,params,result,blockSize);
	free(op);
}


extern "C"
__host__ void setupTransfromFactories() {
	printf("Setting up transform factories\n");
	functions::transform::TransformOpFactory<double> *newOpFactory = new functions::transform::TransformOpFactory<double>();
	functions::transform::TransformOpFactory<float> *newOpFactoryFloat = new functions::transform::TransformOpFactory<float>();
	checkCudaErrors(cudaMemcpyToSymbol(doubleTransformFactory, newOpFactory, sizeof(functions::transform::TransformOpFactory<double> )));
	checkCudaErrors(cudaMemcpyToSymbol(floatTransformFactory, newOpFactory, sizeof(functions::transform::TransformOpFactory<float>)));
	delete(newOpFactory);
	delete(newOpFactoryFloat);

}


#endif

#endif /* TRANSFORM_H_ */
