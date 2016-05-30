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

namespace functions {
	namespace broadcast {
		template <typename T>
		class Broadcast;
	}

	namespace transform {
		template <typename T>
		class Transform;
	}

	namespace reduce {
		template <typename T>
		class ReduceFunction;
	}
}

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



	template<typename T>
	class Im2col {
	public:
		static constexpr const bool requiresSpecial = true;
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)

#endif
		static int outSize(int size, int k, int s, int p, bool coverAll) {
			if (coverAll)
				return (size + p * 2 - k + s - 1) / s + 1;
			else
				return (size + p * 2 - k) / s + 1;
		}

#ifdef __CUDACC__
		/**
		* Based on:  https://github.com/pjreddie/darknet/blob/master/src/im2col_kernels.cu
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
			int kernelWidth = (int)extraParams[0];
			int kernelHeight = (int)extraParams[1];
			int strideX = (int)extraParams[2];
			int strideY = (int)extraParams[3];
			int padWidth = (int)extraParams[4];
			int padHeight = (int)extraParams[5];
			int kSize = kernelWidth * kernelHeight;

			int *outShape = shape::shapeOf(resultShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int samples = inShape[0];
			int depth = inShape[1];
			int height = inShape[2];
			int width = inShape[3];


			int strideex = inStride[0];
			int stridech = inStride[1];
			int strideh = inStride[2];
			int stridew = inStride[3];

			// (height + 2 * padHeight - kernelHeight) / strideX + 1; //
			// (width + 2 * padWidth - kernelWidth) / strideY + 1; //
			int height_col = outShape[4];
			int width_col = outShape[5];

			int n = samples * depth * height_col * width_col;
			/*
			if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, height, width, depth, n, samples);
			*/

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			for (; index < n; index += blockDim.x*gridDim.x) {
				int h_index = index / width_col;
				int h_col = h_index % height_col;
				int w_col = index % width_col;

				int c_im = h_index / height_col;
				int c_col = c_im * kSize;

				int depth_im = c_im % depth;
				int num_im = c_im / depth;
				int h_offset = h_col * strideY - padHeight;
				int w_offset = w_col * strideX - padWidth;

				T* data_col_ptr = result;

				int i_c = (c_col * height_col + h_col) * width_col + w_col;
				data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

				T* data_im_ptr = dx;

				data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

				for (int i = 0; i < kernelHeight; ++i) {
					for (int j = 0; j < kernelWidth; ++j) {
						int h_im = h_offset + i;
						int w_im = w_offset + j;
						int i_f = 0;
						int i_c_temp = i_c;
						for (int dim = 5; dim >= 0; dim--)
						{
							i_f += (i_c_temp % outShape[dim])  * outStride[dim];
							i_c_temp = i_c_temp / outShape[dim];
						}
						result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
						data_col_ptr += height_col * width_col;
						i_c += height_col * width_col;
					}
				}
			}
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
			int kernelWidth = (int)extraParams[0];
			int kernelHeight = (int)extraParams[1];
			int strideX = (int)extraParams[2];
			int strideY = (int)extraParams[3];
			int padWidth = (int)extraParams[4];
			int padHeight = (int)extraParams[5];
			bool coverAll = extraParams[6] > 0.0;

			int outArrayOffset = 0;
			int *outShape = shape::shapeOf(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int inArrayOffset = 0;
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);


			int exampleFrom = 0;
			int exampleTo = inShape[0];
			int depthFrom = 0;
			int depthTo = inShape[1];
			int yOutFrom = 0;
			int yOutTo = outSize(inShape[2], kernelHeight, strideY, padHeight, coverAll);
			int xOutFrom = 0;
			int xOutTo = outSize(inShape[3], kernelWidth, strideX, padWidth, coverAll);


			int *outIndices = new int[6];
			int *inIndices = new int[4];

			int inStride2 = inStride[2];
			int inStride3 = inStride[3];
			int outStride2 = outStride[2];
			int outStride3 = outStride[3];
			int inShape2 = inShape[2];
			int inShape3 = inShape[3];

			bool padding = padHeight > 0 || padWidth > 0;

			T *dIn = dx;
			T *dOut = result;
			//#pragma omp parallel for collapse(2)
			for (int ex = exampleFrom; ex < exampleTo; ex++) {
				for (int d = depthFrom; d < depthTo; d++) {
					inIndices[0] = ex;
					inIndices[1] = d;
					outIndices[0] = ex;
					outIndices[1] = d;

					for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
						for (int y = yOutFrom; y < yOutTo; y++) {  //along height
							outIndices[4] = y;
							outIndices[5] = x;
							int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride,
								outIndices);

							if (padding) {
								int i = y * strideY -
									padHeight;    //index along height of first element of patch in original img
								int j = x * strideX -
									padWidth;     //index along width of first element in patch in original img
								inIndices[2] = i;   //along height
								inIndices[3] = j;   //along width

								int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride,
									inIndices);
								if (outStride2 <= outStride3) {
									//Want dimension 2 (along height) in inner loop for cache reasons
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										int outBufferIdxX = baseOffsetOut + patchX * outStride3;
										int inBufferIdxX = baseOffsetIn + patchX * inStride3;
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 ||
												j + patchX >= inShape3)
												dOut[outBufferIdxX + patchY * outStride2] = 0; //padding
											else {
												dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
													patchY *
													inStride2];
											}
										}
									}
								}
								else {
									//Want dimension 3 in inner loop for cache reasons
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										int outBufferIdxY = baseOffsetOut + patchY * outStride2;
										int inBufferIdxY = baseOffsetIn + patchY * inStride2;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] ||
												j + patchX >= inShape[3])
												dOut[outBufferIdxY + patchX * outStride3] = 0.0; //padding
											else {
												dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
													patchX *
													inStride3];
											}
										}
									}
								}
							}
							else {
								//No padding
								int i = y *
									strideY;    //index along height of first element of patch in original img
								int j = x *
									strideX;     //index along width of first element in patch in original img
								inIndices[2] = i;   //along height
								inIndices[3] = j;   //along width

								int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride,
									inIndices);
								if (outStride2 <= outStride3) {
									//Want dimension 2 (along height) in inner loop for cache reasons
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										int outBufferIdxX = baseOffsetOut + patchX * outStride3;
										int inBufferIdxX = baseOffsetIn + patchX * inStride3;
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
												patchY * inStride2];
										}
									}
								}
								else {
									//Want dimension 3 in inner loop for cache reasons
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										int outBufferIdxY = baseOffsetOut + patchY * outStride2;
										int inBufferIdxY = baseOffsetIn + patchY * inStride2;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
												patchX * inStride3];
										}
									}
								}
							}
						}
					}
				}
			}

			delete[] inIndices;
			delete[] outIndices;

		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[2] != 1) offset += indices[2] * stride[2];
			if (shape[3] != 1) offset += indices[3] * stride[3];
			return offset;
		}


		/**
		* A version of Shape.getOffset without checking on input for negative indices etc
		* normally negative indices are bad, OK here because of other checks on input indices
		* Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};

	template<typename T>
	class Col2Im {

	public:
		static constexpr const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		* https://github.com/pjreddie/darknet/blob/master/src/col2im_kernels.cu
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int strideex = inStride[0];
			int stridech = inStride[1];
			int stridekrow = inStride[2];
			int stridekcol = inStride[3];
			int striderow = inStride[4];
			int stridecol = inStride[5];

			int kernelHeight = inShape[2];
			int kernelWidth = inShape[3];

			// C

			int strideX = (int)extraParams[0];
			int strideY = (int)extraParams[1];
			int padWidth = (int)extraParams[2];
			int padHeight = (int)extraParams[3];
			int imgHeight = (int)extraParams[4];
			int imgWidth = (int)extraParams[5];

			int *outShape = shape::shapeOf(resultShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int samples = outShape[0];
			int depth = outShape[1];
			//int height = outShape[2];
			//int width = outShape[3];

			int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
			int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

			int n = samples * depth * imgHeight * imgWidth;

			/*if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, imgHeight, imgWidth, depth, n, samples);*/



			for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
				T val = 0;
				int w_im = i % imgWidth + padWidth;
				int h_im = (i / imgWidth) % imgHeight + padHeight;
				int c_im = i / (imgWidth * imgWidth);

				int num_im = c_im / depth;
				int depth_im = c_im % depth;

				// compute the start and end of the output
				int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideX + 1;
				int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

				int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideY + 1;
				int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


				for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
					for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
						int h_k = (h_im - h_col * strideY);
						int w_k = (w_im - w_col * strideX);

						int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;

						val += dx[data_col_index];
					}
				}
				int i_f = 0;
				int i_c = i;
				for (int dim = 3; dim >= 0; dim--)
				{
					i_f += (i_c % outShape[dim])  * outStride[dim];
					i_c = i_c / outShape[dim];
				}
				result[i_f] += val;
			}
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {
			int inOffset = 0;
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int kernelHeight = inShape[2];
			int kernelWidth = inShape[3];
			/* int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth, */
			int strideX = (int)extraParams[0];
			int strideY = (int)extraParams[1];
			int padWidth = (int)extraParams[2];
			int padHeight = (int)extraParams[3];


			int exampleFrom = 0;
			int exampleTo = inShape[0];
			int depthFrom = 0;
			int depthTo = inShape[1];

			int outArrayOffset = 0;
			int *outShape = shape::shapeOf(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);


			int *outIndices = new int[4];
			int *inIndices = new int[6];

			int inStride2 = inStride[2];
			int inStride3 = inStride[3];
			int outStride2 = outStride[2];
			int outStride3 = outStride[3];
			int outShape2 = outShape[2];
			int outShape3 = outShape[3];

			int yOutTo = inShape[4];
			int xOutTo = inShape[5];


			bool padding = padHeight > 0 || padWidth > 0;

			T *fIn = dx;
			T *fOut = result;
			//#pragma omp parallel for collapse(2)
			for (int ex = exampleFrom; ex < exampleTo; ex++) {
				for (int d = depthFrom; d < depthTo; d++) {
					inIndices[0] = ex;
					inIndices[1] = d;
					outIndices[0] = ex;
					outIndices[1] = d;

					for (int x = 0; x < xOutTo; x++) {  //Patch number along width
						for (int y = 0; y < yOutTo; y++) {  //Patch number along height
							inIndices[4] = y;   //patch number (along height)
							inIndices[5] = x;   //patch number (along width)
							int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

							if (padding) {
								int i = y * strideY -
									padHeight;    //index along height of first element of patch in original img
								int j = x * strideX -
									padWidth;     //index along width of first element in patch in original img
								outIndices[2] = i;  //along height
								outIndices[3] = j;  //along width

								int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride,
									outIndices);

								if (inStride2 <= inStride3) {
									//Want dimension 2 (along height) in inner loop for cache efficiency
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										if (j + patchX < 0 || j + patchX >= outShape3)
											continue;

										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											if (i + patchY < 0 || i + patchY >= outShape2)
												continue;
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
								else {
									//Want dimension 3 (along width) in inner loop for cache efficiency
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										if (i + patchY < 0 || i + patchY >= outShape2)
											continue;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											if (j + patchX < 0 || j + patchX >= outShape3)
												continue;
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
							}
							else {
								//No padding
								int i = y *
									strideY;    //index along height of first element of patch in output img
								int j = x *
									strideX;     //index along width of first element in patch in output img

								outIndices[2] = i;
								outIndices[3] = j;

								int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride,
									outIndices);

								if (inStride2 <= inStride3) {
									//Want dimension 2 (along height) in inner loop for cache efficiency
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
								else {
									//Want dimension 3 (along width) in inner loop for cache efficiency
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
							}
						}
					}
				}
			}


			delete[] outIndices;
			delete[] inIndices;
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[2] != 1) offset += indices[2] * stride[2];
			if (shape[3] != 1) offset += indices[3] * stride[3];
			return offset;
		}

		/** A version of Shape.getOffset without checking on input for negative indices etc
		* normally negative indices are bad, OK here because of other checks on input indices
		* Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};



	template<typename T>
	class SoftMax {
	public:
		static constexpr const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

			int *shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ int *maxResultShapeBuffer;
			__shared__ functions::reduce::ops::Max<T> *max;
			__shared__ functions::transform::ops::Exp<T> *exp;
			__shared__ functions::broadcast::ops::Subtract<T> *sub;
			__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
			__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
			__shared__ functions::broadcast::ops::Divide<T> *div;
			__shared__ functions::reduce::ops::Sum<T> *sum;
			__shared__ int isVector;

			int length = shape::length(xShapeBuffer);

			if (threadIdx.x == 0) {
				isVector = shape::isVector(xShapeBuffer);
				//maxResult = (T *) allocationPointer + 8; // new T[shape[0]];
				//printf("Launching special SoftMax, shape[0]: [%i]\n", shape[0]);
				maxResult = (T) 0.0;
			}
			__syncthreads();

			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int *stride = shape::stride(xShapeBuffer);
			//iterate along rows
			int dimension[1] = { 0 };
			int maxDimension[1] = { 1 };
			//compute the row wise maxes

			int maxShape[2] = { shape[0], 1 };

			// it's always 2d here
			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			if (threadIdx.x == 0)
				max = new(manager->getFactorySpace()) functions::reduce::ops::Max<T>();
			__syncthreads();

			max->execScalarCuda(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			if (threadIdx.x == 0)
				scalarSub = new(manager->getFactorySpace()) functions::scalar::ops::Subtract<T>();
			__syncthreads();

			scalarSub->transformCuda(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();


			if (threadIdx.x == 0)
				exp = new(manager->getFactorySpace())functions::transform::ops::Exp<T>();
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			exp->transformCuda(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				sum = new(manager->getFactorySpace())functions::reduce::ops::Sum<T>();
			__syncthreads();

			//take the sum for the exponential
			sum->execScalarCuda(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			if (threadIdx.x == 0)
				scalarDiv = new(manager->getFactorySpace())functions::scalar::ops::Divide<T>();
			__syncthreads();

			scalarDiv->transformCuda(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {
			if (shape::isMatrix(xShapeBuffer)) {
				int *shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;
				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr);

				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer)) {
				T max = 0;
				T sum = 0;
				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride >= 1 && resultElementWiseStride >= 1) {
					if (elementWiseStride == 1 && resultElementWiseStride == 1) {
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i]);
						}


						for (int i = 0; i < length; i++) {
							result[i] = dx[i] - max;
						}

						for (int i = 0; i < length; i++) {
							result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						}


						for (int i = 0; i < length; i++) {
							sum += result[i];
						}


						for (int i = 0; i < length; i++) {
							result[i] /= sum;
						}


					}
					else {

						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i * elementWiseStride]);
						}
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = dx[i * elementWiseStride] - max;
						}
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = nd4j::math::nd4j_exp<T>(
								result[i * resultElementWiseStride]);
						}
						for (int i = 0; i < length; i++) {
							sum += result[i * resultElementWiseStride];
						}
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] /= sum;
						}
					}

				}


			}
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};



	template<typename T>
	class LogSoftMax {
	public:
		static constexpr const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		*
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			int *shape = shape::shapeOf(xShapeBuffer);
			int *stride = shape::stride(xShapeBuffer);
			//iterate along rows
			int dimension[1] = { 0 };
			int maxDimension[1] = { 1 };
			__shared__ functions::reduce::ops::Max<T> *max;
			__shared__ functions::transform::ops::Exp<T> *exp;
			__shared__ functions::transform::ops::Log<T> *log;
			__shared__ functions::reduce::ops::Sum<T> *sum;
			__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
			__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
			__shared__ T maxResult;
			__shared__ int isVector;
			__shared__ int *maxResultShapeBuffer;
			if (threadIdx.x == 0) {
				isVector = shape::isVector(xShapeBuffer);

				maxResult = (T) 0.0;
			}
			__syncthreads();
			//compute the row wise maxes

			int maxShape[2] = { shape[0], 1 };
			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			if (threadIdx.x == 0)
				max = new(manager->getFactorySpace()) functions::reduce::ops::Max<T>();
			__syncthreads();

			max->execScalarCuda(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			if (threadIdx.x == 0)
				scalarSub = new(manager->getFactorySpace()) functions::scalar::ops::Subtract<T>();
			__syncthreads();

			scalarSub->transformCuda(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				exp = new(manager->getFactorySpace())functions::transform::ops::Exp<T>();
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			exp->transformCuda(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				sum = new(manager->getFactorySpace())functions::reduce::ops::Sum<T>();
			__syncthreads();

			//take the sum for the exponential
			sum->execScalarCuda(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			if (threadIdx.x == 0)
				scalarDiv = new(manager->getFactorySpace())functions::scalar::ops::Divide<T>();
			__syncthreads();

			scalarDiv->transformCuda(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				log = new functions::transform::ops::Log<T>();
			__syncthreads();

			log->transformCuda(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {

			if (shape::isMatrix(xShapeBuffer, 2)) {
				int *shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;
				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr);

				functions::transform::Transform<T>::template exec<simdOps::Log>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);



				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				T max = 0;
				T sum = 0;

				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {
#pragma omp parallel for simd reduction(max:max) shared(result)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp parallel for simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
						result[i] = nd4j::math::nd4j_log<T>(result[i]);
					}
				}
				else {
#pragma omp parallel for simd reduction(max:max) shared(result, elementWiseStride)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result, elementWiseStride)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp parallel for simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
						result[i * elementWiseStride] = nd4j::math::nd4j_log<T>(result[i * elementWiseStride]);
					}
				}
			}
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};


	/**
	* softmax(x)
	*/
	template<typename T>
	class SoftMaxDerivative {
	public:
		static constexpr const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {


			int *shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ int *maxResultShapeBuffer;
			__shared__ int resultEWS;
			__shared__ functions::reduce::ops::Max<T> *max;
			__shared__ functions::transform::ops::Exp<T> *exp;
			__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
			__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
			__shared__ functions::reduce::ops::Sum<T> *sum;
			__shared__ int isVector;

			int length = shape::length(xShapeBuffer);

			if (threadIdx.x == 0) {
				isVector = shape::isVector(xShapeBuffer);
				resultEWS = shape::elementWiseStride(resultShapeBuffer);

				maxResult = (T) 0.0;
			}
			__syncthreads();

			int *stride = shape::stride(xShapeBuffer);
			//iterate along rows
			int dimension[1] = { 0 };
			int maxDimension[1] = { 1 };
			//compute the row wise maxes

			int maxShape[2] = { shape[0], 1 };

			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			if (threadIdx.x == 0)
				max = new(manager->getFactorySpace()) functions::reduce::ops::Max<T>();
			__syncthreads();


			max->execScalarCuda(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			if (threadIdx.x == 0) delete max;
			__syncthreads();

			//subtract max of each row
			if (threadIdx.x == 0)
				scalarSub = new(manager->getFactorySpace()) functions::scalar::ops::Subtract<T>();
			__syncthreads();

			scalarSub->transformCuda(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				exp = new(manager->getFactorySpace())functions::transform::ops::Exp<T>();
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			exp->transformCuda(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			if (threadIdx.x == 0)
				sum = new(manager->getFactorySpace())functions::reduce::ops::Sum<T>();
			__syncthreads();

			//take the sum for the exponential
			sum->execScalarCuda(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			if (threadIdx.x == 0)
				scalarDiv = new(manager->getFactorySpace())functions::scalar::ops::Divide<T>();
			__syncthreads();

			scalarDiv->transformCuda(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			if (resultEWS >= 1) {
				for (int i = threadIdx.x; i < length; i += blockDim.x) {
					result[i * resultEWS] = result[i * resultEWS] * (1 - result[i * resultEWS]);
				}
			}
			else {
				printf("Non element wise stride not supported right now\n");
			}
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {
			if (shape::isMatrix(xShapeBuffer, 2)) {
				int *shape = shape::shapeOf(xShapeBuffer);

				int resultEleStide = shape::elementWiseStride(resultShapeBuffer);

				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				int len = shape::length(xShapeBuffer);
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);
#pragma omp simd
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;
				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension,
					1, nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1, nullptr, nullptr);

				if (resultEleStide >= 1) {
					if (resultEleStide == 1) {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i] = result[i] * (1 - result[i]);
						}

					}
					else {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i * resultEleStide] = result[i * resultEleStide] * (1 - result[i * resultEleStide]);
						}

					}
				}

				else {
					printf("Non element wise stride not supported right now\n");
				}


				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				T max = 0;
				T sum = 0;

				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {

#pragma omp parallel for simd reduction(max:max) shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp parallel for simd schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
					}

				}
				else {

#pragma omp parallel for simd reduction(max:max) shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}


#pragma omp parallel for simd reduction(+:sum) shared(result, elementWiseStride) schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp parallel for simd schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
					}

				}
			}
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};


	template<typename T>
	class IsMax {
	public:
		static constexpr const bool requiresSpecial = true;


#ifdef __CUDACC__

		inline  __device__ void doAllCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

			__shared__ functions::indexreduce::ops::IMax<T> *max;
			__shared__ int maxIdx;
			__shared__ int length;
			if (threadIdx.x == 0) {
				max = new functions::indexreduce::ops::IMax<T>();
				length = shape::length(resultShapeBuffer);
			}
			__syncthreads();

			max->transform(
				dx,
				xShapeBuffer,
				extraParams,
				result,
				resultShapeBuffer,
				nullptr,
				1,
				1, allocationPointer, reductionPointer, manager, nullptr, nullptr);

			__syncthreads();
			if (threadIdx.x == 0)
				maxIdx = (int)result[0];
			__syncthreads();

			for (int i = threadIdx.x; i < length; i += blockDim.x)
				result[i] = 0;
			__syncthreads();

			if (threadIdx.x == 0) {
				result[maxIdx] = 1.0;

				delete max;
			}

		}
#endif

#ifdef __CUDACC__
		inline __host__

#elif defined(__GNUC__)


#endif
		static void doAll(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {

			int length = shape::length(xShapeBuffer);
			int eleStride = shape::elementWiseStride(xShapeBuffer);
			int resultEleStride = shape::elementWiseStride(resultShapeBuffer);
			char xOrder = shape::order(xShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
			if (xOrder == resultOrder && xOrder == 'c') {
				if (eleStride == 1 && resultEleStride == 1) {
					if (length < 8000) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd
						for (int i = 0; i < length; i++) {
							if (currMax < dx[i]) {
								currMax = dx[i];
								maxIdx = i;
							}

							result[i] = 0.0;

						}

						result[maxIdx] = 1.0;

					}
					else {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp parallel for shared(maxIdx,currMax) schedule(guided)
						for (int i = 0; i < length; i++) {
							if (currMax < dx[i]) {
								currMax = dx[i];
								maxIdx = i;
							}
							result[i] = 0.0;

						}

						result[maxIdx] = 1.0;
					}

				}
				else {
					if (length < 8000) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = 0.0;
							if (currMax < dx[i * eleStride]) {
								currMax = dx[i * eleStride];
								maxIdx = i;
							}
						}

						result[maxIdx * resultEleStride] = 1.0;

					}
					else {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp parallel for shared(maxIdx,currMax) schedule(guided)
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = 0.0;
							if (currMax < dx[i * eleStride]) {
								currMax = dx[i * eleStride];
								maxIdx = i;
							}
						}

						result[maxIdx * resultEleStride] = 1.0;
					}

				}
			}


			else {
				int shapeIter[MAX_RANK];
				int coord[MAX_RANK];
				int dim;
				int xStridesIter[MAX_RANK];
				int resultStridesIter[MAX_RANK];
				int *xShape = shape::shapeOf(xShapeBuffer);
				int *xStride = shape::stride(xShapeBuffer);
				int *resultStride = shape::stride(resultShapeBuffer);
				int rank = shape::rank(xShapeBuffer);
				T *originalResult = result;
				if (PrepareTwoRawArrayIter<T>(rank,
					xShape,
					dx,
					xStride,
					result,
					resultStride,
					&rank,
					shapeIter,
					&dx,
					xStridesIter,
					&result,
					resultStridesIter) >= 0) {
					T value = dx[0];
					int idx = 0;
					int maxIdx = 0;
					ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
						if (dx[0] > value) {
							value = dx[0];
							maxIdx = idx;
						}

						idx++;
						result[0] = 0.0;

					}
					ND4J_RAW_ITER_TWO_NEXT(
						dim,
						rank,
						coord,
						shapeIter,
						dx,
						xStridesIter,
						result,
						resultStridesIter);

					//pointer to where max value would be
					if (shape::order(resultShapeBuffer) == 'c' || (shape::order(resultShapeBuffer) == 'f' &&
						maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1] >=
						shape::length(resultShapeBuffer)))
						originalResult[maxIdx] = 1.0;
					else
						originalResult[maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1]] = 1.0;
				}
			}


		}
	public:


#ifdef __CUDACC__
		/**
		*
		*/

		virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			if (extraParams == nullptr || extraParams[0] == MAX_DIMENSION) {
				this->doAllCuda(dx, xShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, reductionPointer, manager);
			}
			else {
				__shared__ functions::indexreduce::ops::IMax<T> *max;
				__shared__ int maxIdx;
				__shared__ int length;
				if (threadIdx.x == 0) {
					max = new functions::indexreduce::ops::IMax<T>();
					length = shape::length(resultShapeBuffer);
				}

				__syncthreads();

				int dimensionLength = (int)extraParams[0];
				__shared__ int *dimension;
				if (threadIdx.x == 0) {
					dimension = (int *)malloc(sizeof(int) * dimensionLength);
					for (int i = 0; i < dimensionLength; i++) {
						dimension[i] = (int)extraParams[i + 1];
					}
				}

				__syncthreads();

				max->transform(
					dx,
					xShapeBuffer,
					extraParams,
					result,
					resultShapeBuffer,
					dimension,
					dimensionLength,
					1, allocationPointer, reductionPointer, manager, nullptr, nullptr);

				__syncthreads();
				if (threadIdx.x == 0) {
					maxIdx = (int)result[0];
				}
				__syncthreads();

				for (int i = threadIdx.x; i < length; i += blockDim.x)
					result[i] = 0;
				__syncthreads();

				if (threadIdx.x == 0) {
					result[maxIdx] = 1.0;

					delete[] dimension;
					delete max;
				}
			}
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {

			if (extraParams == nullptr || extraParams[0] == 0 ||
				(extraParams[0] == 1 && extraParams[1] == MAX_DIMENSION)) {
				doAll(dx, xShapeBuffer, result, resultShapeBuffer, extraParams);
			}
			else if (shape::isVector(xShapeBuffer)) {
				int dimensionLength = (int)extraParams[0];
				int *dimension = new int[dimensionLength];
				int length = shape::length(xShapeBuffer);
				for (int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int)extraParams[i + 1];
				}
				if (shape::shapeOf(xShapeBuffer)[dimension[0]] == 1) {
					for (int i = 0; i < length; i++) {
						result[i] = 1.0;
					}
				}
				else {
					int eleStride = shape::elementWiseStride(xShapeBuffer);
					if (eleStride == 1) {
						int maxIdx = 0;
						T currMax = dx[0];
						if (length < 8000) {
#pragma omp simd
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i]) {
									currMax = dx[i];
									maxIdx = i;
								}

								dx[i] = 0.0;

							}
						}
						else {

#pragma omp parallel for simd shared(maxIdx,currMax) schedule(guided)
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i]) {
									currMax = dx[i];
									maxIdx = i;
								}

								result[i] = 0.0;

							}
						}

						result[maxIdx] = 1.0;

					}


					else {
						int maxIdx = 0;
						T currMax = dx[0];
						if (length < 8000) {
#pragma omp simd
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i * eleStride]) {
									currMax = dx[i * eleStride];
									maxIdx = i;
								}

								dx[i] = 0.0;

							}
						}
						else {
#pragma omp parallel for simd shared(maxIdx,currMax) schedule(guided)
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i * eleStride]) {
									currMax = dx[i * eleStride];
									maxIdx = i;
								}

								result[i] = 0.0;

							}
						}

						result[maxIdx] = 1.0;

					}
				}


			}
			else {
				int dimensionLength = (int)extraParams[0];
				int *dimension = (int *)malloc(sizeof(int) *dimensionLength);
				for (int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int)extraParams[i + 1];
				}



				shape::TAD tad(xShapeBuffer, dimension, dimensionLength);
				tad.createTadOnlyShapeInfo();
				tad.createOffsets();

				int tads = tad.numTads;
				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
#pragma omp  parallel  for
				for (int i = 0; i < tads; i++) {
					int offset = tad.tadOffsets[i];
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];
					int resultStridesIter[MAX_RANK];
					int *xShape = shape::shapeOf(tadShapeShapeInfo);
					int *xStride = shape::stride(tadShapeShapeInfo);
					int *resultStride = shape::stride(tadShapeShapeInfo);
					int rank = shape::rank(tadShapeShapeInfo);
					T *xPointer = dx + offset;
					T *resultPointer = result + offset;
					T maxValue = xPointer[0];

					T *maxCursor = resultPointer;
					Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
					if (PrepareTwoRawArrayIter<T>(rank,
						xShape,
						xPointer,
						xStride,
						resultPointer,
						resultStride,
						&rank,
						shapeIter,
						&xPointer,
						xStridesIter,
						&resultPointer,
						resultStridesIter) >= 0) {
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							if (maxValue < xPointer[0]) {
								maxCursor = resultPointer;
								maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
								maxValue = xPointer[0];
							}

							resultPointer[0] = 0.0;
						}
						ND4J_RAW_ITER_TWO_NEXT(dim,
							rank,
							coord,
							shapeIter,
							xPointer,
							xStridesIter,
							resultPointer,
							resultStridesIter);
						maxCursor = reinterpret_cast<T *>(maxCursorLong);
						maxCursor[0] = 1.0;
					}
				}

			}
		}

#pragma omp declare simd uniform(params)
		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};
}