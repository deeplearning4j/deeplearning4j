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

#ifdef __JNI__
#include <jni.h>
#endif
namespace functions {
namespace transform {

template<typename T>
class Transform: public  functions::ops::Op<T> {
public:

	/**
	 * The op for transforms
	 * @param d1
	 * @param params
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	inline __device__ __host__

#elif defined(__GNUC__)
	__always_inline

#endif
	T op(T d1, T *params) = 0;

#ifdef __CUDACC__
	__inline__ __device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;
		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i += totalThreads) {
			result[i * incy] = op(dy[i * incy], params);
		}

	}
#endif

	/**
	 * CPU execution
	 * @param dx the input
	 * @param xStride the stride to iterate for the input
	 * @param result the result buffer
	 * @param resultStride the stride for result
	 * storage
	 * @param extraParams the extra parameters
	 * @param n the number of elements to iterate on
	 */
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
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	Transform() {
	}



};

namespace ops {
/**
 * abs(x)
 */
template<typename T>
class Abs: public  Transform<T> {
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

};

/**
 * cei(x)
 */
template<typename T>
class Ceiling: public  Transform<T> {
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

};

/**
 * cos(x)
 */
template<typename T>
class Cosine: public  Transform<T> {
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

};

/**
 * exp(x)
 */
template<typename T>
class Exp: public  Transform<T> {
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

};

/**
 * floor(x)
 */
template<typename T>
class Floor: public  Transform<T> {
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

};

/**
 * log(x)
 */
template<typename T>
class Log: public  Transform<T> {
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

};

/**
 * -x
 */
template<typename T>
class Neg: public  Transform<T> {
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

};

/**
 * pow(x,extra params [0])
 */
template<typename T>
class Pow: public  Transform<T> {
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

/**
 * round(x)
 */
template<typename T>
class Round: public  Transform<T> {
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

};

/**
 * sigmoid(x)
 */
template<typename T>
class Sigmoid: public  Transform<T> {
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

};

/**
 * Scale to be between a
 * min and max
 */
template<typename T>
class SetRange: public  Transform<T> {
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

};

/**
 * sin(x)
 */
template<typename T>
class Sin: public  Transform<T> {
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

};

/**
 * sqrt(x)
 */
template<typename T>
class Sqrt: public  Transform<T> {
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

};

/**
 * softplus(x)
 */
template<typename T>
class SoftPlus: public  Transform<T> {
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


};

/**
 * sign(x)
 */
template<typename T>
class Sign: public  Transform<T> {
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
};

/**
 * tanh(x)
 */
template<typename T>
class Tanh: public  Transform<T> {
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

};

/**
 * acos(x)
 */
template<typename T>
class ACos: public  Transform<T> {
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

};

/**
 * asin(x)
 */
template<typename T>
class ASin: public  Transform<T> {
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

};

/**
 * atan(x)
 */
template<typename T>
class ATan: public  Transform<T> {
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



/**
 * Create an op
 * @param op the op to create
 * 0: abs
 * 1: ceiling
 * 2: cosine
 * 3: exp
 * 4: floor
 * 5: log
 * 6: neg
 * 7: pow
 * 8: round
 * 9: setrange
 * 10:sigmoid
 * 11: sign
 * 12: sin
 * 13:softplus
 * 14:sqrt
 * 15:tanh
 * 16:acos
 * 17:asin
 * 18:atan
 * @return the op given the nnumber
 */
#ifdef __CUDACC__
	__inline__ __device__ __host__
#endif
	Transform<T> * getOp(int op) {
		//gets stuck on string comparison
		Transform<T> *ret = NULL;
		/**
		 * We are likely going to need constant symbols for device memory for different operations
		 * or switch to arithmetic based approaches?
		 */
		if (op == 0) {
			ret =  new transform::ops::Abs<T>();
		}
		else if (op == 1) {
			ret = new transform::ops::Ceiling<T> ();
		}
		if (op == 2) {
			ret = new transform::ops::Cosine<T>();
		}
		else if (op == 3) {
			ret = new transform::ops::Exp<T>();
		}
		else if (op == 4) {
			ret = new transform::ops::Floor<T>();
		}
		else if (op == 5) {
			ret = new transform::ops::Log<T>();
		}
		else if (op == 6) {
			ret = new transform::ops::Neg<T>();
		}
		else if (op == 7) {
			ret = new transform::ops::Pow<T>();
		}
		else if (op == 8) {
			ret =  new transform::ops::Round<T>();
		}
		else if (op == 9) {
			ret = new transform::ops::SetRange<T>();
		}
		else if (op == 10) {
			ret =  new transform::ops::Sigmoid<T>();
		}
		else if (op == 11) {
			ret = new transform::ops::Sign<T>();
		}
		else if (op == 12) {
			ret = new transform::ops::Sin<T>();
		}
		else if (op == 13) {
			ret = new transform::ops::SoftPlus<T>();
		}
		else if (op == 14) {
			ret = new transform::ops::Sqrt<T>();
		}
		else if (op == 15) {
			ret = new transform::ops::Tanh<T>();
		}
		else if (op == 16) {
			ret = new transform::ops::ACos<T> ();
		}
		else if (op == 17) {
			ret = new transform::ops::ASin<T>();
		}
		else if (op == 18) {
			ret = new transform::ops::ATan<T>();
		}
		return ret;
	}

};

}

}

#ifdef __CUDACC__

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		int n,
		int idx,
		T *dy,
		int incy,
		T *params,
		T *result, int blockSize) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;
	if(threadIdx.x == 0) {
		doubleTransformFactory = new functions::transform::TransformOpFactory<T>();

	}

	__syncthreads();


	if(threadIdx.x == 0) {
		op = doubleTransformFactory->getOp(opNum);
	}
	__syncthreads();


	op->transform(n,idx,dy,incy,params,result,blockSize);
	if(threadIdx.x == 0) {
		free(op);
		free(doubleTransformFactory);
	}
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformDouble(
		int opNum,
		int n,
		int idx,
		double *dy,
		int incy,
		double *params,
		double *result, int blockSize) {

	transformGeneric<double>(opNum,n,idx,dy,incy,params,result,blockSize);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformFloat(
		int opNum,
		int n,
		int idx,
		float *dy,
		int incy,
		float *params,
		float *result, int blockSize) {

	transformGeneric<float>(opNum,n,idx,dy,incy,params,result,blockSize);

}

#endif

#endif /* TRANSFORM_H_ */
