#pragma once
#ifndef OPS_H_
#define OPS_H_

#include <helpers/shape.h>
#include <vector>
#include <Environment.h>

#define MIN 1e-12
#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#define MAX_INT 2147483647
#define MIN_CUTFOFF -3.79297773665f
#define FLOAT_MIN_NORMAL 1.17549435e-38
#define FLOAT_MAX_VALUE 3.4028235E38
#define EPS 1e-5
#define AFFINITY close
#ifndef M_E
#define M_E 2.718281828459
#endif

#define no_op_exec_special 	static const bool requiresSpecial = false; static void execSpecial(T *dx, int *xShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets) {}
#define no_op_exec_special_accumulation 	static const bool requiresSpecialAccumulation = false; static void execSpecial(T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffset){}
#ifdef __CUDACC__
#define meta_def __noinline__ __device__
#include <helpers/sharedmem.h>
#define no_op_exec_special_cuda static __device__ void execSpecialCuda(T *dx,int *xShapeBuffer,T *result,int *resultShapeBuffer,T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets) {}
#define no_op_exec_special_accumulation_cuda 	static inline __device__ void execSpecialCuda(T *dx, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfo, int *dimension, int dimensionLength, T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {}
#else
// hacky fix for isnan/being being out of scope
#define isnan std::isnan
#define isinf std::isinf

#define meta_def inline
#define no_op_exec_special_cuda
#define no_op_exec_special_accumulation_cuda
#endif

#ifdef __CUDACC__
#define op_def inline __device__ __host__
#define op_def_special inline __device__

// 610 is for tests only
// 600 is Tesla P100
// 530 is Tegra
#if __CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 530
#define NATIVE_HALFS
#endif

#elif _MSC_VER
#define op_def __pragma("omp declare simd") inline
#define op_def_special __pragma("omp declare simd") inline
#elif __clang__
#define op_def inline
#define op_def_special inline
#elif __GNUC__
#define op_def _Pragma("omp declare simd") inline __attribute__((always_inline))
#define op_def_special _Pragma("omp declare simd") inline __attribute__((always_inline))
#endif

#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_LAMBDA 1.0507009873554804934193349852946

#ifdef _OPENMP
#pragma omp declare reduction(maxT : float,double,float16 :              \
                omp_out = nd4j::math::nd4j_max(omp_in, omp_out) )\
                initializer (omp_priv=-MAX_FLOAT)

#pragma omp declare reduction(minT : float,double,float16 :              \
                omp_out = nd4j::math::nd4j_min(omp_in, omp_out) )\
                initializer (omp_priv=MAX_FLOAT)

#pragma omp declare reduction(sumT : float,double,float16 :              \
                omp_out = omp_in + omp_out)\
                initializer (omp_priv=0.0f)
#endif


namespace functions {
	namespace indexreduce {
		template<typename T>
		struct IndexValue {
			T value;
            Nd4jIndex index;
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return d1 + params[0];
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return d1 - params[0];
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return params[0] - d1;
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return d1 * params[0];
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return d1 / params[0];
		}
	};

    template<typename T>
    class FloorDiv {
    public:
        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_floor<T>(d1 / d2);
        }

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_floor<T>(d1 / d2);
        }

        op_def static T op(T d1) {
            return nd4j::math::nd4j_floor<T>(d1);
        }

        // op for MetaOps
        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_floor<T>(d1 / params[0]);
        }
    };

    template<typename T>
    class TruncateDiv {
    public:
        op_def static T op(T d1, T d2) {
            int i1 = (int) d1;
            int i2 = (int) d2;
            return (T)(i1 / i2);
        }

        op_def static T op(T d1, T d2, T *params) {
            int i1 = (int) d1;
            int i2 = (int) d2;
            return (T)(i1 / i2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps
        op_def static T op(T d1, T *params) {
            int i1 = (int) d1;
            int i2 = (int) params[0];
            return (T)(i1 / i2);
        }
    };

    template<typename T>
    class Remainder {
    public:
        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_remainder(d1, d2);
        }

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_remainder(d1, d2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps
        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_remainder(d1, params[0]);
        }
    };

    template<typename T>
    class FMod {
    public:
        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_fmod(d1, d2);
        }

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_fmod(d1, d2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps
        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_fmod(d1, params[0]);
        }
    };

	template<typename T>
    class FloorMod {
    public:
        op_def static T op(T d1, T d2) {
			T m = nd4j::math::nd4j_fmod(d1, d2);;
            return (d1 < (T) 0.0f) == (d2 < (T) 0.0f) ? m : nd4j::math::nd4j_fmod(m + d2, d2);
        }

        op_def static T op(T d1, T d2, T *params) {
            T m = nd4j::math::nd4j_fmod(d1, d2);
			return (d1 < (T) 0.0f) == (d2 < (T) 0.0f) ? m : nd4j::math::nd4j_fmod(m + d2, d2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps 
        op_def static T op(T d1, T *params) {
			T m = nd4j::math::nd4j_fmod(d1, params[0]);
            return (d1 < (T) 0.0f) == (params[0] < (T) 0.0f) ? m : nd4j::math::nd4j_fmod(m + params[0], params[0]);
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return params[0] / d1;
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return params[0];
		}
	};

	template<typename T>
	class Copy2 {
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

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return params[0];
		}
	};

	template<typename T>
	class Axpy {
	public:
		op_def static T op(T d1, T d2) {
			return d2 + d1;
		}

		op_def static T op(T d1, T d2, T *params) {
			T alpha = params[0];
			return alpha * d1 + d2;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class And {
	public:
		op_def static T op(T d1, T d2) {
			return d2 + d1;
		}

		op_def static T op(T d1, T d2, T *params) {
			T comp = params[0];

			return d1 != comp && d2 != comp ? (T) 1.0f : (T) 0.0f;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Or {
	public:
		op_def static T op(T d1, T d2) {
			return d2 + d1;
		}

		op_def static T op(T d1, T d2, T *params) {
			T comp = params[0];

			return d1 != comp || d2 != comp ? (T) 1.0f : (T) 0.0f;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Xor {
	public:
		op_def static T op(T d1, T d2) {
			return d2 + d1;
		}

		op_def static T op(T d1, T d2, T *params) {
			T comp = params[0];

			return ((d1 == comp && d2 != comp)||(d1 != comp && d2 == comp)) ? (T) 1.0f : (T) 0.0f;
		}

		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Not {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T comp = params[0];

			return d1 == comp ? (T) 1.0f : (T) 0.0f;
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
		/*

		 // just a optional note, feel free to remove later

		op_def static half op(half d1, half d2, half *params) {
			return __float2half(simdOps::Mod<float>::op(__half2float(d1), __half2float(d2), nullptr));
		}
		 */

		op_def static T op(T d1, T d2, T *params) {
			return (int)d1 % (int)d2;
		}

		// op for MetaOp
		op_def static T op(T d1, T *params) {
			return (int)d1 % (int)params[0];
		}
	};

	template<typename T>
	class ReverseMod {
	public:
		op_def static T op(T d1, T d2, T *params) {
			return (int)d2 % (int)d1;
		}

		// op for MetaOp
		op_def static T op(T d1, T *params) {
			return (int)params[0] % (int)d1;
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
			T absDiff = nd4j::math::nd4j_abs<T>(diff);
			if (absDiff <= (T) MIN)
				return (T) 1.0f;
			return (T) 0.0f;
		}

		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class EqualTo {
	public:
		op_def static T op(T d1, T d2) {
			return d1 == d2;
		}

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
		op_def static T op(T d1, T d2) {
			return d1 != d2;
		}

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
		op_def static T op(T d1, T d2) {
			return d1 >= d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 >= d2;
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static T op(T d1, T *params) {
			return d1;
		}
	};


	template<typename T>
	class GreaterThan {
	public:
		op_def static T op(T d1, T d2) {
			return d1 > d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return d1 > d2;
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static T op(T d1, T *params) {
			return d1;
		}

	};


	template<typename T>
	class LessThan {
	public:
		op_def static T op(T d1, T d2) {
			return d1 < d2;
		}

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
		op_def static T op(T d1, T d2) {
			return d1 <= d2;
		}

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
			return ((d1 >= (T)-1.0 && d1 <= (T) 1.0) ? 1.0 : 0.0);
		}
	};

	
	template<typename T>
	class HardTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			if (d1 < (T) -1.0) return -1.0;
			else if (d1 > (T) 1.0) return 1.0;
			else return d1;
			//return d1 < -1.0 ? -1.0 : d1 > 1.0 ? 1.0 : d1;
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
	class Log1p {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_log<T>(1+d1);
		}
	};

	template<typename T>
	class LogX {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_log<T>(d1) / nd4j::math::nd4j_log<T>(params[0]) ;
		}
	};

    template<typename T>
    class StabilizeFP16 {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            if (d1 <= (T) 0.) return 0.001;
                else return d1;
        }
    };

	template<typename T>
	class SpecialDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * ((T) 1.0 - d1);
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
	class Erf {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_erf<T>(d1);
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

		op_def static T op(T d1, T d2) {
			return nd4j::math::nd4j_pow<T>(d1, d2);
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_pow<T>(d1, d2);
		}

		op_def static T op(T d1) {
			return d1;
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
	class IsNan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return isnan(d1) ? (T) 1.0f : (T) 0.0f;
		}
	};

	template<typename T>
	class IsInf {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return isinf(d1) ? (T) 1.0f : (T) 0.0f;
		}
	};

	template<typename T>
	class IsFinite {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (!isinf(d1) && ! isnan(d1))? (T) 1.0f : (T) 0.0f;
		}
	};


	template<typename T>
	class ClipByValue {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			if (d1 > params[1])
				return params[1];
			else if (d1 < params[0])
				return params[0];
			else return d1;
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
	class Swish {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * nd4j::math::nd4j_sigmoid<T>(d1);
		}
	};


	template<typename T>
	class SwishDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T ex = nd4j::math::nd4j_pow<T>(M_E, d1);
			return (ex * (d1 + ex + 1)) / nd4j::math::nd4j_pow<T>((ex + 1) , (T)2.0f);
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

    template<typename T>
    class HardSigmoid {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_min<T>((T) 1.0, nd4j::math::nd4j_max<T>((T) 0.0f, ((T) 0.2f) * d1 + (T) 0.5f));
        }
    };

    template<typename T>
    class HardSigmoidDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 < (T) -2.5f || d1 > (T) 2.5f ? (T) 0.0f : (T) 0.2f;
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
			if (min == (T) 0.0f && max == (T) 1.0f) {
				T val = (T) 1.0f / ((T) 1.0f + nd4j::math::nd4j_exp<T>(-d1));
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
	class RSqrt {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (T) 1.0f / nd4j::math::nd4j_sqrt<T>(d1);
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
			return (d1 > (T) 0.0f) - (d1 < (T) 0.0f);
		}
	};


	template<typename T>
	class TimesOneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * ((T) 1.0 - d1);
		}
	};


	template<typename T>
	class RationalTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			// keep 2/3 as runtime variable, to match precision
			T dis = ((T) 2.0f / (T) 3.0f) * d1;

			T tanh = nd4j::math::nd4j_sgn<T>(dis) * ((T) 1.0f - ((T) 1.0f / ((T) 1.0f + nd4j::math::nd4j_abs<T>(dis) + nd4j::math::nd4j_pow<T>(dis, 2) + (T) 1.41645f * nd4j::math::nd4j_pow<T>(dis, 4) )));
			return (T) 1.7159f * tanh;
		}
	};

	template<typename T>
	class RationalTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T dis = ((T) 2.0f / (T) 3.0f) * d1;

			T a = (T) 1.0f + nd4j::math::nd4j_abs<T>(dis) + nd4j::math::nd4j_pow<T>(dis, 2) + (T) 1.41645f * nd4j::math::nd4j_pow<T>(dis,4);

			T tDeriv = ((T)1.0f + nd4j::math::nd4j_sign<T>(dis) * ((T) 2.0f * dis + (T) 4.0f * (T) 1.41645f * nd4j::math::nd4j_pow<T>(dis, 3))) / (a * a);

			return (T) 1.7159f * ((T) 2.0f / (T) 3.0f) * tDeriv;
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
    class RectifiedTanh {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_max<T>((T) 0.0f, nd4j::math::nd4j_tanh<T>(d1));
        }
    };

    template<typename T>
    class RectifiedTanhDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 > (T) 0.0f ? nd4j::math::nd4j_tanhderivative<T>(d1) : (T) 0.0f;
        }
    };

	template<typename T>
	class ATanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_atanh<T>(d1);
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
	class Cube {
	public:
		no_op_exec_special
			no_op_exec_special_cuda

			op_def static T op(T d1, T *params) {
			return d1 * d1 * d1;
		}
	};


	template<typename T>
	class CubeDerivative {
	public:
		no_op_exec_special
			no_op_exec_special_cuda

			op_def static T op(T d1, T *params) {
			return 3 * d1 * d1;
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
			return (T) 1.0f;
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
    class MatchCondition {
    public:
		no_op_exec_special
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return old + opOutput;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return old + opOutput;
        }

        // this op return 1.0 if condition met, 0.0 otherwise
        op_def static T op(T d1, T *extraParams) {
            T compare = extraParams[0];
            T eps = extraParams[1];

            int mode = (int) extraParams[2];


            // printf("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

            if (mode == 0) // equals
                return nd4j::math::nd4j_abs<T>(d1 - compare) <= eps ? 1.0 : 0.0;
            else if (mode == 1) // not equals
                return nd4j::math::nd4j_abs<T>(d1 - compare) > eps ? 1.0 : 0.0;
            else if (mode == 2) // less_than
                return d1 < compare? 1.0 : 0.0;
            else if (mode ==3) // greater_than
                return d1 > compare? 1.0 : 0.0;
            else if (mode == 4) // less_or_equals_than
                return d1 <= compare? 1.0 : 0.0;
            else if (mode == 5) // greater_or_equals_than
                return d1 >= compare? 1.0 : 0.0;
            else if (mode == 6) // abs_less_than
                return nd4j::math::nd4j_abs<T>(d1) < compare? 1.0 : 0.0;
            else if (mode == 7) // abs_greater_than
                return nd4j::math::nd4j_abs<T>(d1) > compare? 1.0 : 0.0;
            else if (mode == 8) // is inf
                return isinf(d1) ? 1.0 : 0.0;
            else if (mode == 9) // is nan
                return isnan(d1) ? 1.0 : 0.0;
            else if (mode == 10)
                return (d1 == compare) ? 1.0 : 0.0;
            else if (mode == 11)
                return (d1 != compare) ? 1.0 : 0.0;
			else if (mode == 12) // abs_greater_or_equals_than
				return nd4j::math::nd4j_abs<T>(d1) >= compare? 1.0 : 0.0;
			else if (mode == 13) // abs_less_or_equals_than
				return nd4j::math::nd4j_abs<T>(d1) <= compare? 1.0 : 0.0;
            else
                printf("Undefined match condition: [%i]\n", mode);

            return d1;
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return reduction;
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
    class SELU {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 > (T) 0.0f ? (T) SELU_LAMBDA * d1 : (T) SELU_LAMBDA * ((T) SELU_ALPHA * nd4j::math::nd4j_exp<T>(d1) - (T) SELU_ALPHA);
        }
    };

    template<typename T>
    class SELUDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 > (T) 0.0f ? (T) SELU_LAMBDA : (T) SELU_ALPHA * (T) SELU_LAMBDA * nd4j::math::nd4j_exp<T>(d1);
        }
    };

	template<typename T>
	class LeakyRELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			if (d1 >= (T) 0.0f) return (T) 1.0f;
			else return params[0];
			//return (d1 >= (T) 0.0 ? 1.0 : params[0]);
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
	class Sinh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_sinh<T>(d1);
		}
	};

	template<typename T>
	class Cosh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_cosh<T>(d1);
		}
	};


	template<typename T>
	class Tan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_tan<T>(d1);
		}
	};

    template<typename T>
    class TanDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return  (T) 1.0f / (T) nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_cos<T>(d1), (T) 2.0f);
        }
    };

    template<typename T>
    class ASinh {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_asinh<T>(d1);
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
    class Atan2 {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_atan2<T>(d2, d1);
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
			if (d1 * k > (T) - MIN_CUTFOFF)
				return (T)((T)- MIN_CUTFOFF / k);
			else if (d1 * k < (T) MIN_CUTFOFF)
				return (T)((T) MIN_CUTFOFF / k);
			return d1;
		}
	};



	template<typename T>
	class Step {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (d1 > params[0] ? (T) 1.0f : (T) 0.0f);
		}
	};



	template<typename T>
	class OneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (T) 1.0f - d1;
		}
	};

	template<typename T>
	class Sum {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
    class ShannonEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_pow<T>(d1, (T) 2.0f) * nd4j::math::nd4j_log<T>(nd4j::math::nd4j_pow<T>(d1, (T) 2.0f));
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return -reduction;
        }
    };


    template<typename T>
    class LogEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_log<T>(nd4j::math::nd4j_pow<T>(d1, (T) 2.0f));
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return reduction;
        }
    };

    template<typename T>
    class Entropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return d1 * nd4j::math::nd4j_log<T>(d1);
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return reduction;
        }
    };


    template<typename T>
    class ASum {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(opOutput) + nd4j::math::nd4j_abs<T>(old);
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(opOutput) + nd4j::math::nd4j_abs<T>(old);
        }

        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(d1);
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(reduction);
        }
    };


	template<typename T>
    class CountNonZero {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return d1 == (T) 0.0f ? (T) 0.0f : (T) 1.0f;
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return reduction;
        }
    };


	template<typename T>
	class Prod {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 1.0f;
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
	class Any {
	public:
		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
			return reduction > (T) 0.0f ? (T) 1.0f : (T) 0.0f ;
		}
	};


    template<typename T>
    class All {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 1.0f;
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
            return reduction > (T) 0.0f ? (T) 1.0f : (T) 0.0f ;
        }
    };

	template<typename T>
	class Mean {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
			return reduction / (int) n;
		}
	};


    template<typename T>
    class AMean {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(opOutput) + nd4j::math::nd4j_abs<T>(old);
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(opOutput) + nd4j::math::nd4j_abs<T>(old);
        }

        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(d1);
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(reduction) / (int) n;
        }
    };

	template<typename T>
	class Max {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

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

        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_max<T>(d1, d2);
        }

		// FIXME: this signature overlaps with MetaOp
		op_def static T op(T d1, T *extraParams) {
			return d1;
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return reduction;
		}
	};


    template<typename T>
    class AMax {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return input[0];
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(old), nd4j::math::nd4j_abs<T>(opOutput));
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(opOutput), nd4j::math::nd4j_abs<T>(old));
        }

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(d1), nd4j::math::nd4j_abs<T>(d2));
        }

        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_abs<T>(d1) > nd4j::math::nd4j_abs<T>(d2) ? d1 : d2;
        }

        // FIXME: this signature overlaps with MetaOp
        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(d1);
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(reduction);
        }
    };


	template<typename T>
	class AMin {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return input[0];
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(nd4j::math::nd4j_abs<T>(old), nd4j::math::nd4j_abs<T>(opOutput));
		}

		op_def static T update(T old, T opOutput, T *extraParams) {
			return nd4j::math::nd4j_min<T>(nd4j::math::nd4j_abs<T>(opOutput), nd4j::math::nd4j_abs<T>(old));
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_min(nd4j::math::nd4j_abs<T>(d1), nd4j::math::nd4j_abs<T>(d2));
		}

        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_abs<T>(d1) < nd4j::math::nd4j_abs<T>(d2) ? d1 : d2;
        }

		// FIXME: this signature overlaps with MetaOp
		op_def static T op(T d1, T *extraParams) {
			return nd4j::math::nd4j_abs<T>(d1);
		}

		op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
			return nd4j::math::nd4j_abs<T>(reduction);
		}
	};

    template<typename T>
    class Min {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

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

        op_def static T op(T d1, T d2) {
            return nd4j::math::nd4j_min(d1, d2);
        }

        // FIXME: this signature overlaps with MetaOp
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
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
			return (reduction - (nd4j::math::nd4j_pow<T>(bias, (T) 2.0f) / (int) n))
                / (n - (int) 1);
		}
	};

	/**
	* Standard deviation of a buffer
	*/
	template<typename T>
	class StandardDeviation {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return (T) 0.0f;
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
			return (T) 0.0f;
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
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],(T) (d1 * d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],(T) (d2 * d2));

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


    template<typename T>
    class JaccardDistance {
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
            return (T) 0.0f;
        }

        op_def static  T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            // num / denom
            return ((T) 1.0f) - (extraParams[0] / extraParams[1]);
        }

        op_def static T num(T d1, T d2) {
            return nd4j::math::nd4j_min<T>(d1, d2);
        }

        op_def static T denom(T d1, T d2) {
            return nd4j::math::nd4j_max<T>(d1, d2);
        }

        op_def static T op(T d1, T d2, T *extraParams) {
            extraParams[0] += num(d1, d2);
            extraParams[1] += denom(d1, d2);
            return (T) 0.0f;
        }

        op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
        __device__
		static inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],(T) num(d1, d2));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],(T) denom(d1, d2));

			return (T) 0.0f;
		}
#endif

        op_def static  T update(T old, T opOutput, T *extraParams) {
            return old + opOutput;
        }


        op_def static T merge(T old, T opOutput, T *extraParams) {
            return update(old, opOutput, extraParams);
        }
    };


    template<typename T>
    class SimpleHammingDistance {
    public:
        static const int extraParamsLen = 0;

        op_def static T *generateExtraParams() {
            //T *extraParams = new T[2];
            return nullptr;
        }

        op_def static void finalizeExtraParams(T *extraParams) {
            //delete[] extraParams;
        }

        op_def static T startingValue(T *input) {
            return (T) 0.0f;
        }

        op_def static  T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return (T) (reduction / (T) n);
        }

        op_def static T op(T d1, T d2, T *extraParams) {
            return (d1 == d2) ? (T) 0.0f :  (T)1.0f;
        }

        op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {

        }

#ifdef __CUDACC__
        __device__
		static inline T opAtomic(T d1, T d2, T *extraParams) {
			return op(d1, d2, extraParams);
		}
#endif

        op_def static  T update(T old, T opOutput, T *extraParams) {
            return old + opOutput;
        }


        op_def static T merge(T old, T opOutput, T *extraParams) {
            return update(old, opOutput, extraParams);
        }
    };

    template<typename T>
    class CosineDistance {
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
            return (T) 0.0f;
        }

        op_def static  T postProcess(T reduction, Nd4jIndex n, T *extraParams) {
            return ((T) 1.0f) - (reduction / (nd4j::math::nd4j_sqrt<T>(extraParams[0]) * nd4j::math::nd4j_sqrt<T>(extraParams[1])));
        }

        op_def static T op(T d1, T d2, T *extraParams) {
            extraParams[0] += nd4j::math::nd4j_abs<T>(d1) * nd4j::math::nd4j_abs<T>(d1);
            extraParams[1] += nd4j::math::nd4j_abs<T>(d2) * nd4j::math::nd4j_abs<T>(d2);
            return (d1 * d2);
        }

        op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
        __device__
		static inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],(T) nd4j::math::nd4j_abs<T>(d1) * nd4j::math::nd4j_abs<T>(d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],(T) nd4j::math::nd4j_abs<T>(d2) * nd4j::math::nd4j_abs<T>(d2));

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
			return (T) 0.0f;
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


    /**
	* Op to check equality within arrays
	*/
    template<typename T>
    class EqualsWithEps {
    public:
        static const int extraParamsLen = 0;

        op_def static T * generateExtraParams() {
            return nullptr;
        }

        op_def static void finalizeExtraParams(T *extraParamsRef) {
            //no-op
        }

        op_def static T startingValue(T *input) {
            return (T) 0.0f;
        }

        op_def static T postProcess(T reduction, Nd4jIndex n, T *extraParamsRef) {
            return reduction;
        }

        op_def static T op(T d1, T d2, T *extraParamsRef) {
            T abs1 = nd4j::math::nd4j_abs<T>(d1);
            T abs2 = nd4j::math::nd4j_abs<T>(d2);
            T diff = nd4j::math::nd4j_abs<T>(d1 - d2);
            T eps = extraParamsRef[2];

            if (d1 == d2) {
                return (T) 0.0f;
            } else if (d1 == (T) 0.0f || d2 == (T) 0.0f || diff < (T) FLOAT_MIN_NORMAL) {
                //if (eps > 0.1)
                return diff < eps ? 0.0f : 1.0f;

                //return diff <  (T) (eps * FLOAT_MIN_NORMAL) ? 0.0f : 1.0f;
                //return res;
            } else {
                T xDiff = (diff / nd4j::math::nd4j_min<T>((abs1 + abs2), FLOAT_MAX_VALUE));
                return  xDiff < eps ? (T) 0.0f : (T) 1.0f;
            }
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
			return (T) 0.0f;
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
			return (T) 0.0f;
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

		op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {

		}


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
	class IndexAbsoluteMax  {
	public:
#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return nd4j::math::nd4j_abs<T>(val);
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> update(
				functions::indexreduce::IndexValue<T> old,
		functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
			opOutput.value = nd4j::math::nd4j_abs<T>(opOutput.value);
			old.value = nd4j::math::nd4j_abs<T>(old.value);
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

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
		functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (nd4j::math::nd4j_abs<T>(f1.value) > nd4j::math::nd4j_abs<T>(f2.value))
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline T startingValue(T *input) {
			return MIN_FLOAT;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
		functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};

    template<typename T>
    class FirstIndex {
    public:
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> val, T *extraParams) {
            return val;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<T> update(
                functions::indexreduce::IndexValue<T> old,
                functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {

#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            T res = simdOps::MatchCondition<T>::op(opOutput.value, extraParams);

			//printf("res: %f; oldIdx: %i; newIdx: %i\n", res, old.index, opOutput.index);

            if (res == (T) 0.0f)
                return old;

            if (old.index < 0)
                return opOutput;

            if (old.index > opOutput.index)
                return opOutput;

            return old;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T startingValue(T *input) {
            return -MAX_FLOAT;
        }


#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
                                                               functions::indexreduce::IndexValue<T> d2, T *extraParams) {
            return d1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> merge(
                functions::indexreduce::IndexValue<T> f1,
                functions::indexreduce::IndexValue<T> f2, T *extraParams) {
            if (f1.index > f2.index)
                return f2;
            return f1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> postProcess(
                functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
                T *dx, int incx, T *extraParams, T *result) {
            return reduction;
        }
    };


    template<typename T>
    class LastIndex {
    public:
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> val, T *extraParams) {
            return val;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<T> update(
                functions::indexreduce::IndexValue<T> old,
                functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            T res = simdOps::MatchCondition<T>::op(opOutput.value, extraParams);

            if (res == (T) 0.0f)
                return old;

            if (old.index < 0)
                return opOutput;

            if (old.index < opOutput.index)
                return opOutput;

            return old;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T startingValue(T *input) {
            return -MAX_FLOAT;
        }


#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
                                                               functions::indexreduce::IndexValue<T> d2, T *extraParams) {
            return d1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> merge(
                functions::indexreduce::IndexValue<T> f1,
                functions::indexreduce::IndexValue<T> f2, T *extraParams) {
            if (f1.index < f2.index)
                return f2;
            return f1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> postProcess(
                functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
                T *dx, int incx, T *extraParams, T *result) {
            return reduction;
        }
    };


	template<typename T>
	class IndexMax  {
	public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<T> update(
				functions::indexreduce::IndexValue<T> old,
				functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
			if (opOutput.value > old.value) {
                return opOutput;
            }
#ifdef __CUDACC__
			// workaround for cuda race condition at merge phase
			else if (opOutput.value == old.value && opOutput.index < old.index)
				return opOutput;
#elif defined(__GNUC__)

#endif
			return old;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
				functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (f1.value > f2.value)
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T startingValue(T *input) {
			return -MAX_FLOAT;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
				functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};


	template<typename T>
	class IndexAbsoluteMin {
	public:
#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> op(
				functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline T startingValue(T *input) {
			return MAX_FLOAT;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> update(
				functions::indexreduce::IndexValue<T> old,
		functions::indexreduce::IndexValue<T> opOutput, T *extraParams) {
			opOutput.value = nd4j::math::nd4j_abs<T>(opOutput.value);
			old.value = nd4j::math::nd4j_abs<T>(old.value);
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

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
		functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (nd4j::math::nd4j_abs<T>(f1.value) < nd4j::math::nd4j_abs<T>(f2.value))
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
		functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};


	template<typename T>
	class IndexMin {
	public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(
				functions::indexreduce::IndexValue<T> val, T *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T startingValue(T *input) {
			return MAX_FLOAT;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> startingIndexValue(T *input) {
            functions::indexreduce::IndexValue<T> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> update(
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


#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> merge(
				functions::indexreduce::IndexValue<T> f1,
				functions::indexreduce::IndexValue<T> f2, T *extraParams) {
			if (f1.value < f2.value)
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> postProcess(
				functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
				T *dx, int incx, T *extraParams, T *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
				functions::indexreduce::IndexValue<T> d2, T *extraParams) {
			return d1;
		}
	};

	template<typename T>
	class SummaryStatsVariance {
	public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < (T) 0.0f)
					return val.variance();
				return ret;
			}
			return val.variance();
		}
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

	template<typename T>
	class SummaryStatsStandardDeviation {
	public:
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < (T) 0.0f)
					return nd4j::math::nd4j_sqrt(val.variance());
				else
					return nd4j::math::nd4j_sqrt(ret);
			}
			return  nd4j::math::nd4j_sqrt(val.variance());
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

template<typename T>
	class DropOut {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#ifdef __CUDACC__
        __device__
#endif
		inline static T op(T d1, T *params) {
			T prob = params[0];

#ifdef __CUDACC__
			T length = params[1];
            T tid = gridDim.x * blockDim.x + threadIdx.x;
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>((T) clock64() * (T) tid + (T) length * (T) tid));
#else
			T rnd = (T) rand() / (T) RAND_MAX;
#endif
			return rnd >= prob ? (T) 0.0f : d1;
		}
	};

template<typename T>
	class DropOutInverted {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#ifdef __CUDACC__
    __device__
#endif
        inline static T op(T d1, T *params) {
			T prob = params[0];
#ifdef __CUDACC__
			T length = params[1];
			T tid = gridDim.x * blockDim.x + threadIdx.x;
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>((T) clock64() * (T) tid + (T) length * (T) tid));
#else
			T rnd = (T) rand() / (T) RAND_MAX;
#endif
			return rnd >= prob ? (T) 0.0f : d1 / prob;
		}
	};


	template<typename T>
	class ReplaceNans {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T replacement = params[0];
			return isnan(d1) ? replacement : d1 ;
		}
	};

    // this op is used for conditional pairwise transforms only
    template<typename T>
    class CompareAndReplace{
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        // op definition for PairWise Transform
        op_def static T op(T d1, T d2, T *params) {
            T compare = params[0];
            T eps = params[2];
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<T>(d1 - compare) <= eps)
                    return d2;
                else
                    return d1;
            else if (mode == 1) // not equals eps
                if (nd4j::math::nd4j_abs<T>(d1 - compare) > eps)
                    return d2;
                else
                    return d1;
            else if (mode == 2) // less_than eps
                if (d1 < compare)
                    return d2;
                else
                    return d1;
            else if (mode ==3) // greater_than
                if (d1 > compare)
                    return d2;
                else
                    return d1;
            else if (mode == 4) // less_or_equals_than
                if (d1 <= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 5) // greater_or_equals_than
                if (d1 >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 6) // abs_less_than
                if (nd4j::math::nd4j_abs<T>(d1) < compare)
                    return d2;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<T>(d1) > compare)
                    return d2;
                else
                    return d1;
            else if (mode == 8) // is inf
                if (isinf(d1))
                    return d2;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (isnan(d1))
                    return d2;
                else
                    return d1;
            else if (mode == 10)
                if (d1 == compare)
                    return d2;
                else
                    return d1;
            else if (mode == 11)
                if (d1 != compare)
                    return d2;
                else
                    return d1;
            else if (mode == 12) // abs_greater_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) <= compare)
                    return d2;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
        }
    };

	template<typename T>
	class CompareAndSet {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

        // op definition for Transform
		op_def static T op(T d1, T *params) {
			T compare = params[0];
			T set = params[1];
			T eps = params[2];

            // with mode == 0 we do set if d1 equals to compare, and with mode == 1 - we go otherwise
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<T>(d1 - compare) <= eps)
                    return set;
				else
                    return d1;
			    //return nd4j::math::nd4j_abs<T>(d1 - compare) <= eps ? set : d1;
            else if (mode == 1) // not equals
                if (nd4j::math::nd4j_abs<T>(d1 - compare) > eps)
                    return set;
                else
                    return d1;
                //return nd4j::math::nd4j_abs<T>(d1 - compare) > eps ? set : d1;
            else if (mode == 2) // less_than
                if (d1 < compare)
                    return set;
                else
                    return d1;
            else if (mode ==3) // greater_than
                if (d1 > compare)
                    return set;
                else
                    return d1;
            else if (mode == 4) // less_or_equals_than
                if (d1 <= compare)
                    return set;
                else
                    return d1;
            else if (mode == 5) // greater_or_equals_than
                if (d1 >= compare)
                    return set;
                else
                    return d1;
            else if (mode == 6) // abs_less_than
                if (nd4j::math::nd4j_abs<T>(d1) < compare)
                    return set;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<T>(d1) > compare)
                    return set;
                else
                    return d1;
            else if (mode == 8) // is inf
                if (isinf(d1))
                    return set;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (isnan(d1))
                    return set;
                else
                    return d1;
            else if (mode == 10)
                if (d1 == compare)
                    return set;
                else
                    return d1;
            else if (mode == 11)
                if (d1 != compare)
                    return set;
                else
                    return d1;
            else if (mode == 12) // abs_greater_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) >= compare)
                    return set;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) <= compare)
                    return set;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
		}

        // op definition for PairWise Transform
        op_def static T op(T d1, T d2, T *params) {
            T compare = params[0];
            T eps = params[2];
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<T>(d2 - compare) <= eps)
                    return d2;
                else
                    return d1;
            else if (mode == 1) // not equals
                if (nd4j::math::nd4j_abs<T>(d2 - compare) > eps)
                    return d2;
                else
                    return d1;
            else if (mode == 2) // less_than
                if (d2 < compare)
                    return d2;
                else
                    return d1;
            else if (mode ==3) // greater_than
                if (d2 > compare)
                    return d2;
                else
                    return d1;
            else if (mode == 4) // less_or_equals_than
                if (d2 <= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 5) // greater_or_equals_than
                if (d2 >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 6) // abs_less_than
                if (nd4j::math::nd4j_abs<T>(d2) < compare)
                    return d2;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<T>(d2) > compare)
                    return d2;
                else
                    return d1;
            else if (mode == 8) // is inf
                if (isinf(d2))
                    return d2;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (isnan(d2))
                    return d2;
                else
                    return d1;
            else if (mode == 10)
                if (d2 == compare)
                    return d2;
                else
                    return d1;
            else if (mode == 11)
                if (d2 != compare)
                    return d2;
                else
                    return d1;
            else if (mode == 12) // abs_greater_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<T>(d1) <= compare)
                    return d2;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
        }
	};

/**
 * Special case here: MetaOp which consist of 2 operations.
 *
 * Predicate can be either scalar or transform, to process data before actual op call
 * Postulate will be the scalar/transform, but will be applied to result of broadcast/reduce/reduce3
 */
template<typename T, typename OpTypeA, typename OpTypeB>
	class MetaOp {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		/*
		 * PREDICATE
		 */

		meta_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

		// scalar, transform, reduce, indexreduce entry
		meta_def static T op(T d1, T *params) {
			/*
			 * We assume, that params for MetaOp is a set of pointers to actual op A & B extraArgs
			 */
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
		}

		// PWT, broadcast entry. Predicate can be only scalar, transform
		meta_def static T op(T d1, T d2, T *params) {
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::op(d1, paramsA), d2, paramsB);
		}

		/*
		 * POSTULATE
		 */

		// will be called for reduce, reduce3
		meta_def static T postProcess(T reduction, Nd4jIndex n, T *params) {
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::postProcess(reduction, n, paramsA), paramsB);
		}

	};

    /**
     * InvertedMetaOp shares the same idea as MetaOp, but op being applied to op.Y in pairwise/broadcast ops
     */
template<typename T, typename OpTypeA, typename OpTypeB>
    class InvertedMetaOp {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        /*
         * PREDICATE
         */

        // scalar, transform, reduce, indexreduce entry
		op_def static T op(T d1, T *params) {
            /*
             * We assume, that this method won't be EVER called
             */
            printf("You should NEVER see this message in output\n");
            return (T) 0.0f;
        }

        // PWT, broadcast entry. Predicate can be only scalar, transform
        op_def static T op(T d1, T d2, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::op(OpTypeA::op(d1, d2, paramsA), paramsB);
        }

        /*
         * POSTULATE
         */

        // will be called for reduce, reduce3
        op_def static T postProcess(T reduction, Nd4jIndex n, T *params) {
            /*
             * We assume, that this method won't be EVER called
             */
            printf("You should NEVER EVER see this message in output\n");

            return (T) 0.0f;
        }

    };


template<typename T, typename OpTypeA, typename OpTypeB>
    class ReduceMetaOp {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

		meta_def static T startingValue(const T *input) {
            return OpTypeB::startingValue(input);
        }

		meta_def static T merge(T old, T opOutput, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
//            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::merge(old, opOutput, paramsB);
        }

		meta_def static T update(T old, T opOutput, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            //T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::update(old, opOutput, paramsB);
        }

		meta_def static T op(T d1, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
        }

		meta_def static T postProcess(T reduction, Nd4jIndex n, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
//            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::postProcess(reduction, n, paramsB);
        }
    };
}

#endif
	