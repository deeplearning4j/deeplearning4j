#pragma once

#include <shape.h>
#include <vector>

#define no_op_exec_special 	static constexpr const bool requiresSpecial = false; static void execSpecial(T *dx, int *xShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams) {}
#define MIN 1e-12

#ifdef __CUDACC__
#define op_def inline __host__  __device__
#define no_op_exec_special_cuda 	static __device__ void execSpecialCuda(T *dx,int *xShapeBuffer,T *result,int *resultShapeBuffer,T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {}
#else
#define op_def inline
#define no_op_exec_special_cuda
#endif


namespace simdOps {
	template<typename T>
	class Add {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 + d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 + d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Subtract {
	public:

#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 - d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 - d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseSubtract {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2 - d1;
		}
		
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d2 - d1;
		}
#pragma omp declare simd		
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Multiply {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 * d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 * d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Divide {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 / d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 / d2;
		}
		
#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseDivide {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2 / d1;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d2 / d1;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Copy {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class SetValOrLess {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			if (d2 < d1) {
				return d1;
			}
			return d2;
		}
	};

	template<typename T>
	class Mod {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return (int)d1 % (int)d2;
		}
	};

	template<typename T>
	class ReverseMod {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return (int)d2 % (int)d1;
		}
	};

	/**
	* Whether 2 elements in an array
	* are epsilion equal
	*/
	template<typename T>
	class Epsilon {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			T diff = d1 - d2;
			T absDiff = nd4j::math::nd4j_abs(diff);
			if (absDiff < MIN)
				return 1;
			return 0;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class EqualTo {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 == d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class NotEqualTo {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 != d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}
	};



	template<typename T>
	class GreaterThanOrEqual {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 >= d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class GreaterThan {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 > d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class LessThan {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 < d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class LessThanOrEqual {
	public:
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return d1 <= d2;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class Abs {
	public:
		no_op_exec_special
		no_op_exec_special_cuda
		
#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_abs<T>(d1);
		}
	};


	template<typename T>
	class Ceiling {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_ceil<T>(d1);
		}
	};

	
	template<typename T>
	class Cosine {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_cos<T>(d1);
		}
	};

	
	template<typename T>
	class Exp {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_exp<T>(d1);
		}
	};

	
	template<typename T>
	class HardTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return ((d1 >= -1.0 && d1 <= 1.0) ? 1.0 : 0.0);
		}
	};

	
	template<typename T>
	class HardTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1 < -1.0 ? -1.0 : d1 > 1.0 ? 1.0 : d1;
		}
	};


	template<typename T>
	class Floor {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_floor<T>(d1);
		}
	};


	template<typename T>
	class Log {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_log<T>(d1);
		}
	};

	template<typename T>
	class SpecialDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1 * (1.0 - d1);
		}
	};


	template<typename T>
	class Neg {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return -d1;
		}
	};


	template<typename T>
	class Pow {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_pow<T>(d1, params[0]);
		}
	};

	
	template<typename T>
	class Round {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_round<T>(d1);
		}
	};

	
	template<typename T>
	class Sigmoid {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sigmoid<T>(d1);
		}
	};



	template<typename T>
	class SigmoidDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sigmoidderivative<T>(d1);
		}
	};


	/**
	* Scale to be between a min and max
	*/
	template<typename T>
	class SetRange {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
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
	};

	
	template<typename T>
	class Sin {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sin<T>(d1);
		}
	};

	
	template<typename T>
	class Sqrt {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sqrt<T>(d1);
		}
	};

	
	template<typename T>
	class SoftPlus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};

	
	template<typename T>
	class Sign {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return (d1 > 0) - (d1 < 0);
		}
	};


	template<typename T>
	class TimesOneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1 * (1 - d1);
		}
	};


	template<typename T>
	class Tanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_tanh<T>(d1);
		}
	};


	template<typename T>
	class TanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_tanhderivative<T>(d1);
		}
	};

	template<typename T>
	class ACos {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_acos<T>(d1);
		}
	};


	template<typename T>
	class Ones {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return 1;
		}
	};


	
	template<typename T>
	class SoftSign {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_softsign<T>(d1);
		}
	};


	template<typename T>
	class SoftSignDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_softsignderivative<T>(d1);
		}
	};

	template<typename T>
	class ELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_elu<T>(d1);
		}
	};


	template<typename T>
	class ELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_eluderivative<T>(d1);
		}
	};


	template<typename T>
	class RELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1 < params[0] ? params[0] : d1;
		}
	};


	template<typename T>
	class LeakyRELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_leakyrelu<T>(d1, params[0]);
		}
	};

	template<typename T>
	class LeakyRELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return (d1 >= 0 ? 1.0 : params[0]);
		}
	};


	template<typename T>
	class ASin {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_asin<T>(d1);
		}
	};

	
	template<typename T>
	class ATan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_atan(d1);
		}
	};

	
	template<typename T>
	class Identity {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}
	};

	
	template<typename T>
	class Stabilize {
	public:
		static const constexpr double realMin = 1.1755e-38f;
		static const constexpr double cutOff = nd4j::math::nd4j_log(realMin);
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			T k = params[0];
			if (d1 * k > -cutOff)
				return (T)(-cutOff / k);
			else if (d1 * k < cutOff)
				return (T)(cutOff / k);
			return d1;
		}
	};



	template<typename T>
	class Step {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return (d1 > params[0] ? 1.0 : 0.0);
		}
	};



	template<typename T>
	class OneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return 1.0 - d1;
		}
	};

	template<typename T>
	class Sum {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return (T) 0.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}
		
#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams, n)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Prod {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return (T) 1.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput * old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput * old;
		}
#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};

	template<typename T>
	class Mean {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			(void)input;
			return 0.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction / (T)n;
		}
	};


	template<typename T>
	class Max { 
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return input[0];
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(old, opOutput);
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(opOutput, old);
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_max<T>(d1, d2);
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Min {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return input[0];
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(old, opOutput);
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(opOutput, old);
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_min(d1, d2);
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};

	
	template<typename T>
	class Norm1 {
	public:
#pragma omp declare simd uniform(input)
			op_def static T startingValue(const T *input) {
			return 0.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}
#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return nd4j::math::nd4j_abs<T>(d1);
		}

#pragma omp declare simd uniform(extraParams, n)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Norm2 {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1 * d1;
		}

#pragma omp declare simd uniform(extraParams, n)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return nd4j::math::nd4j_sqrt<T>(reduction);
		}
	};

	
	template<typename T>
	class NormMax {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}

#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(old),
				nd4j::math::nd4j_abs<T>(opOutput));

		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

#pragma omp declare simd uniform(extraParams)
			op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(reduction),
				nd4j::math::nd4j_abs<T>(reduction));
		}
	};

	template<typename T>
	class Variance {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return 0.0;
		}
#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}
#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			T mean = extraParams[0];
			T ret = d1 - mean;
			return ret * ret;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			T bias = extraParams[1];
			return (reduction - (nd4j::math::nd4j_pow<T>(bias, 2.0) / (T)n))
				/ (T)(n - 1.0);
		}
	};

	/**
	* Standard deviation of a buffer
	*/
	template<typename T>
	class StandardDeviation {
	public:
#pragma omp declare simd uniform(input)
		op_def static T startingValue(const T *input) {
			return 0.0;
		}
#pragma omp declare simd uniform(extraParams)
		op_def static T merge(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}
#pragma omp declare simd uniform(extraParams)
		op_def static T update(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}

#pragma omp declare simd uniform(extraParams)
		op_def static T op(T d1, T *extraParams) {
			T mean = extraParams[0];
			T ret = d1 - mean;
			return ret * ret;
		}

#pragma omp declare simd uniform(extraParams)
		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			T ret = Variance<T>::postProcess(reduction, n, extraParams);
			T sqrtRet = nd4j::math::nd4j_sqrt<T>(ret);
			return sqrtRet;
		}
	};
}