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



#ifdef __CUDACC__
	__device__ __host__
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
__device__ __constant__ functions::transform::TransformOpFactory<double> *doubleTransformFactory;
__device__ __constant__ functions::transform::TransformOpFactory<float> *floatTransformFactory;


extern "C" __global__ void transformDouble(
		int opNum,
		int n,
		int idx,
		double *dy,
		int incy,
		double *params,
		double *result, int blockSize) {

	functions::transform::Transform<double> *op = doubleTransformFactory->getOp(opNum);
	functions::transform::ops::Sigmoid<double> sigmoid;
	printf("Obtained op\n");
	sigmoid.transform(n,idx,dy,incy,params,result,blockSize);
	free(op);
}

extern "C" __global__ void transformFloat(
		int opNum,
		int n,
		int idx,
		float *dy,
		int incy,
		float *params,
		float *result, int blockSize) {

	functions::transform::Transform<float> *op = floatTransformFactory->getOp(opNum);
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
