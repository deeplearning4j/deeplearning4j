#pragma once

#include <shape.h>
#include <vector>

#define MIN 1e-12
#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#define MIN_CUTFOFF -3.79297773665f

#define no_op_exec_special 	static const bool requiresSpecial = false; static void execSpecial(T *dx, int *xShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams) {}
#ifdef __CUDACC__
#include <sharedmem.h>
#define no_op_exec_special_cuda static __device__ void execSpecialCuda(T *dx,int *xShapeBuffer,T *result,int *resultShapeBuffer,T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {}
#else
#define no_op_exec_special_cuda
#endif

#ifdef __CUDACC__
#define op_def inline __device__
#elif _MSC_VER
#define op_def __pragma("omp declare simd") inline
#elif __clang__
#define op_def inline
#elif __GNUC__
#define op_def _Pragma("omp declare simd") inline
#endif


namespace functions {
	namespace indexreduce {
		template<typename T>
		struct IndexValue {
			T value;
			unsigned int index;
		};
	}

	namespace summarystats {
		template <typename T>
		class SummaryStatsData;
	}
}

namespace simdOps {
	template<typename T>
	class Add {
	public:
		op_def static T op(T d1, T d2) {
			return d1 + d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 + d2;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Subtract {
	public:
		op_def static T op(T d1, T d2) {
			return d1 - d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 - d2;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseSubtract {
	public:
		op_def static T op(T d1, T d2) {
			return d2 - d1;
		}
		
		op_def static T op(T d1, T d2, T *params) {
			return d2 - d1;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Multiply {
	public:
		op_def static T op(T d1, T d2) {
			return d1 * d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 * d2;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Divide {
	public:
		op_def static T op(T d1, T d2) {
			return d1 / d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 / d2;
		}
		
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseDivide {
	public:
		op_def static T op(T d1, T d2) {
			return d2 / d1;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d2 / d1;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Copy {
	public:
		op_def static T op(T d1, T d2) {
			return d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d2;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class SetValOrLess {
	public:
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
		op_def static T op(T d1, T d2, T *params) {
			return (int)d1 % (int)d2;
		}
	};

	template<typename T>
	class ReverseMod {
	public:
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
		op_def static T op(T d1, T d2, T *params) {
			T diff = d1 - d2;
			T absDiff = nd4j::math::nd4j_abs(diff);
			if (absDiff < MIN)
				return 1;
			return 0;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class EqualTo {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 == d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class NotEqualTo {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 != d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}
	};



	template<typename T>
	class GreaterThanOrEqual {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 >= d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class GreaterThan {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 > d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class LessThan {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 < d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class LessThanOrEqual {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return d1 <= d2;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class Abs {
	public:
		no_op_exec_special
		no_op_exec_special_cuda
		
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_abs<T>(d1);
		}
	};


	template<typename T>
	class Ceiling {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_ceil<T>(d1);
		}
	};

	
	template<typename T>
	class Cosine {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_cos<T>(d1);
		}
	};

	
	template<typename T>
	class Exp {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_exp<T>(d1);
		}
	};

	
	template<typename T>
	class HardTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return ((d1 >= -1.0 && d1 <= 1.0) ? 1.0 : 0.0);
		}
	};

	
	template<typename T>
	class HardTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 < -1.0 ? -1.0 : d1 > 1.0 ? 1.0 : d1;
		}
	};


	template<typename T>
	class Floor {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_floor<T>(d1);
		}
	};


	template<typename T>
	class Log {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_log<T>(d1);
		}
	};

	template<typename T>
	class SpecialDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * (1.0 - d1);
		}
	};


	template<typename T>
	class Neg {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return -d1;
		}
	};


	template<typename T>
	class Pow {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_pow<T>(d1, params[0]);
		}
	};

	
	template<typename T>
	class Round {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_round<T>(d1);
		}
	};

	
	template<typename T>
	class Sigmoid {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sigmoid<T>(d1);
		}
	};



	template<typename T>
	class SigmoidDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

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

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sin<T>(d1);
		}
	};

	
	template<typename T>
	class Sqrt {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sqrt<T>(d1);
		}
	};

	
	template<typename T>
	class SoftPlus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};

	
	template<typename T>
	class Sign {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (d1 > 0) - (d1 < 0);
		}
	};


	template<typename T>
	class TimesOneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * (1 - d1);
		}
	};


	template<typename T>
	class Tanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_tanh<T>(d1);
		}
	};


	template<typename T>
	class TanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_tanhderivative<T>(d1);
		}
	};

	template<typename T>
	class ACos {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_acos<T>(d1);
		}
	};


	template<typename T>
	class Ones {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return 1;
		}
	};


	
	template<typename T>
	class SoftSign {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_softsign<T>(d1);
		}
	};


	template<typename T>
	class SoftSignDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_softsignderivative<T>(d1);
		}
	};

	template<typename T>
	class ELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_elu<T>(d1);
		}
	};


	template<typename T>
	class ELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_eluderivative<T>(d1);
		}
	};


	template<typename T>
	class RELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 < params[0] ? params[0] : d1;
		}
	};


	template<typename T>
	class LeakyRELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_leakyrelu<T>(d1, params[0]);
		}
	};

	template<typename T>
	class LeakyRELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (d1 >= 0 ? 1.0 : params[0]);
		}
	};


	template<typename T>
	class ASin {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_asin<T>(d1);
		}
	};

	
	template<typename T>
	class ATan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_atan(d1);
		}
	};

	
	template<typename T>
	class Identity {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1;
		}
	};

	
	template<typename T>
	class Stabilize {
	public:
	
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			//const double realMin = 1.1755e-38f;
			//const double cutOff = nd4j::math::nd4j_log(realMin);

			T k = params[0];
			if (d1 * k > - MIN_CUTFOFF)
				return (T)(- MIN_CUTFOFF / k);
			else if (d1 * k < MIN_CUTFOFF)
				return (T)(MIN_CUTFOFF / k);
			return d1;
		}
	};



	template<typename T>
	class Step {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (d1 > params[0] ? 1.0 : 0.0);
		}
	};



	template<typename T>
	class OneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return 1.0 - d1;
		}
	};

	template<typename T>
	class Sum {
	public:
		op_def static T startingValue(const T *input) {
			return (T) 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}
		
		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Prod {
	public:
		op_def static T startingValue(const T *input) {
			return (T) 1.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput * old;
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput * old;
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};

	template<typename T>
	class Mean {
	public:
		op_def static T startingValue(const T *input) {
			(void)input;
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction / (T)n;
		}
	};


	template<typename T>
	class Max { 
	public:
		op_def static T startingValue(const T *input) {
			return input[0];
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(old, opOutput);
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(opOutput, old);
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_max<T>(d1, d2);
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Min {
	public:
		op_def static T startingValue(const T *input) {
			return input[0];
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(old, opOutput);
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(opOutput, old);
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_min(d1, d2);
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};

	
	template<typename T>
	class Norm1 {
	public:
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}

		op_def static T op(T d1, T *extraParams) {
			return nd4j::math::nd4j_abs<T>(d1);
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Norm2 {
	public:
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T op(T d1, T *extraParams) {
			return d1 * d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return nd4j::math::nd4j_sqrt<T>(reduction);
		}
	};

	
	template<typename T>
	class NormMax {
	public:
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;

		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(old),
				nd4j::math::nd4j_abs<T>(opOutput));
		}

		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(reduction),
				nd4j::math::nd4j_abs<T>(reduction));
		}
	};

	template<typename T>
	class Variance {
	public:
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return old + opOutput;
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}

		op_def static T op(T d1, T *extraParams) {
			T mean = extraParams[0];
			T ret = d1 - mean;
			return ret * ret;
		}

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
		op_def static T startingValue(const T *input) {
			return 0.0;
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return old + opOutput;
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return old + opOutput;

		}

		op_def static T op(T d1, T *extraParams) {
			T mean = extraParams[0];
			T ret = d1 - mean;
			return ret * ret;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			T ret = Variance<T>::postProcess(reduction, n, extraParams);
			T sqrtRet = nd4j::math::nd4j_sqrt<T>(ret);
			return sqrtRet;
		}
	};

	template<typename T>
	class CosineSimilarity {
	public:
		static const int extraParamsLen = 2;

		op_def static T *generateExtraParams() {
			//T *extraParams = new T[2];
			return nullptr;
		}

		op_def static void finalizeExtraParams(T *extraParams) {
			//delete[] extraParams;
		}

		op_def static T startingValue(T *input) {
			return 0.0;
		}

		op_def static  T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction / (nd4j::math::nd4j_sqrt<T>(extraParams[0]) * nd4j::math::nd4j_sqrt<T>(extraParams[1]));
		}

		op_def static T op(T d1, T d2, T *extraParams) {
			extraParams[0] += d1 * d1;
			extraParams[1] += d2 * d2;
			return (d1 * d2);
		}

		op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {
			extraParamsTotal[0] += extraParamsLocal[0];
			extraParamsTotal[1] += extraParamsLocal[1];
		}

#ifdef __CUDACC__
		__device__
		static inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0], d1 * d1);
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], d2 * d2);

			return (d1 * d2);
		}
#endif

		op_def static  T update(T old, T opOutput, T *extraParams) {
			return old + opOutput;
		}


		op_def static T merge(T old, T opOutput, T *extraParams) {
			return update(old, opOutput, extraParams);
		}
	};


	/**
	* Dot product between 2 arrays
	*/
	template<typename T>
	class Dot {
	public:
		static const int extraParamsLen = 0;

		op_def static T * generateExtraParams() {
			return nullptr;
		}
		
		op_def static void finalizeExtraParams(T *extraParamsRef) {
			//no-op
			//delete[] * extraParamsRef;
		}
		
		op_def static T startingValue(T *input) {
			return 0.0;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParamsRef) {
			return reduction;
		}

		op_def static T op(T d1, T d2, T *extraParamsRef) {
			return d1 * d2;
		}

	
#ifdef __CUDACC__
		__device__
		static inline T opAtomic(T d1, T d2, T *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static T update(T old, T opOutput, T *extraParamsRef) {
			return opOutput + old;
		}

		op_def static T merge(T old, T opOutput, T *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}

		op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {}
	};



	template<typename T>
	class EuclideanDistance {
	public:
		static const int extraParamsLen = 0;

		op_def static T * generateExtraParams() {
			return nullptr;
		}

		op_def static void finalizeExtraParams(T *extraParamsRef) {
			//no-op
		}

		op_def static T startingValue(T *input) {
			return 0.0;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParamsRef) {
			return nd4j::math::nd4j_sqrt<T>(reduction);
		}
		
		op_def static T op(T d1, T d2, T *extraParamsRef) {
			T ret = d1 - d2;
			return ret * ret;
		}

		
#ifdef __CUDACC__
			__device__
			static  inline T opAtomic(T d1, T d2, T *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static T update(T old, T opOutput, T *extraParamsRef) {
			return opOutput + old;
		}

		op_def static T merge(T old, T opOutput, T *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
		op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {}

	};


	template<typename T>
	class ManhattanDistance  {
	public:
		static const int extraParamsLen = 0;

		op_def static T * generateExtraParams() {
			return nullptr;
	}

		op_def static void finalizeExtraParams(T *extraParamsRef) {
			//no-op
		}

		op_def static T startingValue(T *input) {
			return 0.0;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParamsRef) {
			return reduction;
		}

		op_def static T op(T d1, T d2, T *extraParamsRef) {
			return nd4j::math::nd4j_abs<T>(d1 - d2);
		}

		op_def static  T update(T old, T opOutput, T *extraParamsRef) {
			return old + opOutput;
		}

		op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {}
		

#ifdef __CUDACC__
		__device__
		static inline T opAtomic(T d1, T d2, T *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

#ifndef __clang__
#pragma omp declare simd uniform(extraParamsRef)
#endif
		op_def static T merge(T old, T opOutput, T *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
	};


	template<typename T>
	class IndexMax  {
	public:
		op_def static functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return val;
		}

		op_def static functions::indexreduce::IndexValue<T> update(
				functions::indexreduce::IndexValue<T> old,
				functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
			if (opOutput.value > old.value)
				return opOutput;
#ifdef __CUDACC__
			// workaround for cuda race condition at merge phase
			else if (opOutput.value == old.value && opOutput.index < old.index)
				return opOutput;
#elif defined(__GNUC__)

#endif
			return old;
		}

		op_def static functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
				functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (f1.value > f2.value)
				return f2;
			return f1;
		}


		op_def static functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

		op_def static T startingValue(T *input) {
			return MIN_FLOAT;
		}

		op_def static functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
				functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};


	template<typename T>
	class IndexMin {
	public:
		op_def static functions::indexreduce::IndexValue<T> op(
				functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return val;
		}

		op_def static T startingValue(T *input) {
			return MAX_FLOAT;
		}

		op_def static functions::indexreduce::IndexValue<T> update(
				functions::indexreduce::IndexValue<T> old,
				functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
			if (opOutput.value < old.value)
				return opOutput;

#ifdef __CUDACC__
			// workaround for cuda race condition at merge phase
			else if (opOutput.value == old.value && opOutput.index < old.index)
				return opOutput;
#elif defined(__GNUC__)

#endif
			return old;
		}

		op_def static functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
				functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (f1.value < f2.value)
				return f2;
			return f1;
		}

		op_def static functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

		op_def static functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
				functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};

	template<typename T>
	class SummaryStatsVariance {
	public:
		op_def static T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < 0)
					return val.variance();
				return ret;
			}
			return val.variance();
		}

		op_def static functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

	template<typename T>
	class SummaryStatsStandardDeviation {
	public:
		op_def static T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < 0)
					return nd4j::math::nd4j_sqrt(val.variance());
				else
					return nd4j::math::nd4j_sqrt(ret);
			}
			return  nd4j::math::nd4j_sqrt(val.variance());
		}

		op_def static functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

template<typename T>
	class DropOut {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T prob = params[0];
			T length = params[1];
#ifdef __CUDACC__
            int tid = gridDim.x * blockDim.x + threadIdx.x;
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>(clock64() * tid + length * tid));
#else
			T rnd = (T) rand() / (T) RAND_MAX;
#endif
			return rnd <= prob ? (T) 0.0 : d1;
		}
	};

template<typename T>
	class DropOutInverted {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T prob = params[0];
			T length = params[1];
#ifdef __CUDACC__
			int tid = gridDim.x * blockDim.x + threadIdx.x;
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>(clock64() * tid + length * tid));
#else
			T rnd = (T) rand() / (T) RAND_MAX;
#endif
			return rnd >= prob ? 0 : d1 / prob;
		}
	};
}
