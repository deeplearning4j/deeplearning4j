/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#pragma once
#ifndef OPS_H_
#define OPS_H_

#include <op_boilerplate.h>
#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <vector>
#include <Environment.h>
#include <loops/summarystatsreduce.h>

#define MIN 1e-12
#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#define MAX_INT 2147483647
#define MIN_CUTFOFF -3.79297773665f
#define FLOAT_MIN_NORMAL 1.17549435e-38
#define EPS 1e-5
#define AFFINITY close
#define DOUBLE_PI_T T(2.0 * 3.14159265358979323846)

#define no_op_exec_special 	static const bool requiresSpecial = false; static void execSpecial(T *dx, Nd4jLong *xShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation 	static const bool requiresSpecialAccumulation = false; static void execSpecial(T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
#ifdef __CUDACC__
#include <helpers/sharedmem.h>
#define no_op_exec_special_cuda static __device__ void execSpecialCuda(T *dx, Nd4jLong *xShapeBuffer,T *result, Nd4jLong *resultShapeBuffer,T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation_cuda 	static inline __device__ void execSpecialCuda(T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {}
#else
// hacky fix for isnan/being being out of scope
//#ifdef IOS
//#define isinf(x) 0 // this isn't right. But std::isinf fails
//#define isnan(x) 0
//#else
//#define isnan std::isnan
//#define isinf std::isinf
//#endif

#define no_op_exec_special_cuda
#define no_op_exec_special_accumulation_cuda
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
            Nd4jLong index;
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
		
		op_def static T startingValue() {
			return static_cast<T>(0.f);
		}
	};

	template <typename X, typename Y, typename Z>
	class NewAdd {
    public:
        op_def static Z op(X d1, Y d2, X *params) {
            return d1 + d2;
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
	class SquaredSubtract {
	public:
		op_def static T op(T d1, T d2) {
			return nd4j::math::nd4j_pow<T>(d1 - d2, static_cast<T>(2.f));
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_pow<T>(d1 - d2, static_cast<T>(2.f));
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_pow<T>(d1 - params[0], static_cast<T>(2.f));
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
	class LogPoisonLossFull {

	public:
		op_def static T op(T z, T c) {
			return (nd4j::math::nd4j_exp<T>(c) - z * c  + (z * nd4j::math::nd4j_log<T>(z) - z + static_cast<T>(0.5f) * nd4j::math::nd4j_log<T>(DOUBLE_PI_T * z)));
		}

		op_def static T op(T z, T c, T *params) {
			return (nd4j::math::nd4j_exp<T>(c) - z * c  + (z * nd4j::math::nd4j_log<T>(z) - z + static_cast<T>(0.5f) * nd4j::math::nd4j_log<T>(DOUBLE_PI_T * z)));
		}

		op_def static T op(T z) {
			return (z * nd4j::math::nd4j_log<T>(z) - z + static_cast<T>(0.5f) * nd4j::math::nd4j_log<T>(DOUBLE_PI_T * z));
		}

		// op for MetaOps
		op_def static T op(T z, T *params) {
			return (nd4j::math::nd4j_exp<T>(params[0]) - z * params[0]  + (z * nd4j::math::nd4j_log<T>(z) - z + static_cast<T>(0.5f) * nd4j::math::nd4j_log<T>(DOUBLE_PI_T * z)));
		}
	};

	template<typename T>
	class LogPoisonLoss {

	public:
		op_def static T op(T z, T c) {
			return (nd4j::math::nd4j_exp<T>(c) - z * c);
		}

		op_def static T op(T z, T c, T *params) {
			return (nd4j::math::nd4j_exp<T>(c) - z * c);
		}

		op_def static T op(T z) {
			return (z);
		}

		// op for MetaOps
		op_def static T op(T z, T *params) {
			return (nd4j::math::nd4j_exp<T>(params[0]) - z * params[0]);
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

		op_def static T startingValue() {
			return static_cast<T>(1.f);
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

		op_def static T startingValue() {
			return static_cast<T>(1.f);
		}
	};

	template<typename T>
	class SafeDivide {
	public:
		op_def static T op(T d1, T d2) {
			if(d2 == static_cast<T>(0.f))
				return static_cast<T>(0.f);
			return d1 / d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			if(d2 == static_cast<T>(0.f))
				return static_cast<T>(0.f);
			return d1 / d2;
		}
		
		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			if(params[0] == static_cast<T>(0.f))
				return static_cast<T>(0.f);
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
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<T>(i1 / i2);
        }

        op_def static T op(T d1, T d2, T *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<T>(i1 / i2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps
        op_def static T op(T d1, T *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(params[0]);
            return static_cast<T>(i1 / i2);
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
            return (d1 < static_cast<T>(0.0f)) == (d2 < static_cast<T>(0.0f)) ? m : nd4j::math::nd4j_fmod<T>(m + d2, d2);
        }

        op_def static T op(T d1, T d2, T *params) {
            T m = nd4j::math::nd4j_fmod(d1, d2);
			return (d1 < static_cast<T>(0.0f)) == (d2 < static_cast<T>(0.0f)) ? m : nd4j::math::nd4j_fmod<T>(m + d2, d2);
        }

        op_def static T op(T d1) {
            return d1;
        }

        // op for MetaOps 
        op_def static T op(T d1, T *params) {
			T m = nd4j::math::nd4j_fmod(d1, params[0]);
            return (d1 < static_cast<T>(0.0f)) == (params[0] < static_cast<T>(0.0f)) ? m : nd4j::math::nd4j_fmod<T>(m + params[0], params[0]);
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

			return d1 != comp && d2 != comp ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
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

			return d1 != comp || d2 != comp ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
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

			return ((d1 == comp && d2 != comp)||(d1 != comp && d2 == comp)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
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

			return d1 == comp ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}
	};

	template<typename T>
	class LogicalNot {
	public:
		op_def static T op(T d1, T d2) {
			return !((int) d1  && (int) d2);
		}

		op_def static T op(T d1, T d2, T *params) {
			return (T) !((int) d1  && (int) d2);
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
		}
	};

	template<typename T>
	class LogicalXor {
	public:
		op_def static T op(T d1, T d2) {
		    int i1 = (int) d1;
		    int i2 = (int) d2;

			return  (i1 | i2) &~ (i1 & i2);
		}

		op_def static T op(T d1, T d2, T *params) {
			int i1 = (int) d1;
			int i2 = (int) d2;

			return  (i1 | i2) &~ (i1 & i2);
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
		}
	};

	template<typename T>
	class LogicalAnd {
	public:
		op_def static T op(T d1, T d2) {
			return (int) d1  & (int) d2;
		}

		op_def static T op(T d1, T d2, T *params) {
			return (int) d1  & (int) d2;
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
		}
	};

	template<typename T>
	class LogicalOr {
	public:
		op_def static T op(T d1, T d2) {
			return (int) d1  | (int) d2;
		}

		op_def static T op(T d1, T d2, T *params) {
            return (int) d1  | (int) d2;
		}

		op_def static T op(T d1) {
			return d1;
		}

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return static_cast<T>(119.0f);
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

		op_def static T op(T d1, T d2) {
            return (int)d1 % (int)d2;
        }

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
        op_def static T op(T d1, T d2) {
            return (int)d2 % (int)d1;
        }

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
			if (absDiff <= static_cast<T>(MIN))
				return static_cast<T>(1.0f);
			return static_cast<T>(0.0f);
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
			return ((d1 >= static_cast<T>(-1.0f) && d1 <= static_cast<T>(1.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f));
		}
	};

	
	template<typename T>
	class HardTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			if (d1 < static_cast<T>(-1.0f))
				return static_cast<T>(-1.0f);
			else if (d1 > static_cast<T>(1.0f))
				return static_cast<T>(1.0f);
			else
				return d1;

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
            if (d1 <= static_cast<T>(0.f)) return static_cast<T>(0.001f);
                else return d1;
        }
    };

	template<typename T>
	class SpecialDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * (static_cast<T>(1.0f) - d1);
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
	class Erfc {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_erfc<T>(d1);
		}
	};

	template<typename T>
	class Reciprocal {
	public:
		no_op_exec_special
		no_op_exec_special_cuda
//		op_def static T op(T d1) {
//			return (T(1.0f) / d1);
//		}
		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return (static_cast<T>(1.0f)/d1);
		}
	};

	template<typename T>
	class Sqr {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_pow<T>(d1, static_cast<T>(2.f));
		}

		op_def static T op(T d1) {
			return nd4j::math::nd4j_pow<T>(d1, static_cast<T>(2.0f));
		}
	};


	template<typename T>
	class RelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_re<T>(d1, params[0]);
		}

		op_def static T op(T d1, T d2) {
			return nd4j::math::nd4j_re<T>(d1, d2);
		}

		op_def static T op(T d1, T d2, T *params) {
			return nd4j::math::nd4j_re<T>(d1, d2);
		}

		op_def static T op(T d1) {
			return static_cast<T>(0.0f);
		}
	};

	template<typename T>
	class BinaryRelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T d2 = params[0];
			T threshold = params[1];
			return nd4j::math::nd4j_re<T>(d1, d2) > threshold ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
 		}

		op_def static T op(T d1, T d2, T *params) {
			T threshold = params[0];
			return nd4j::math::nd4j_re<T>(d1, d2) > threshold ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

		op_def static T op(T d1) {
			return static_cast<T>(0.0f);
		}
	};

	template<typename T>
	class BinaryMinimumAbsoluteRelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T d2 = params[0];
			T thresholdRelative = params[1];
			T thresholdAbsolute = params[2];
			return nd4j::math::nd4j_re<T>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<T>(d1 - d2) < thresholdAbsolute ? static_cast<T>(0.0f) : static_cast<T>(1.0f)) : static_cast<T>(0.0f);
 		}

		op_def static T op(T d1, T d2, T *params) {
			T thresholdRelative = params[0];
			T thresholdAbsolute = params[1];
			return nd4j::math::nd4j_re<T>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<T>(d1 - d2) < thresholdAbsolute ? static_cast<T>(0.0f) : static_cast<T>(1.0f)) : static_cast<T>(0.0f);
		}

		op_def static T op(T d1) {
			return static_cast<T>(0.0f);
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
	class PowDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return params[0] * nd4j::math::nd4j_pow<T>(d1, params[0] - static_cast<T>(1.f));
		}

		op_def static T op(T d1, T d2) {
			return d2 * nd4j::math::nd4j_pow<T>(d1, d2 - static_cast<T>(1.f));
		}

		op_def static T op(T d1, T d2, T *params) {
			return d2 * nd4j::math::nd4j_pow<T>(d1, d2 - static_cast<T>(1.f));
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

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_isnan(d1) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }


        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }


        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
        }
	};


	template<typename T>
	class Expm1 {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_exp(d1) - static_cast<T>(1.0f);
		}
	};

	template<typename T>
	class IsInf {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_isinf<T>(d1) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }


        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }


        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
        }
	};


	template<typename T>
	class IsInfOrNan{
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_isfin<T>(d1) ? static_cast<T>(0.0f) : static_cast<T>(1.0f);
		}

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction;
		}
	};



	template<typename T>
	class IsFinite {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_isfin<T>(d1) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
		}

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }


        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
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
			T ex = nd4j::math::nd4j_pow<T>(static_cast<T>(M_E), d1);
			return (ex * (d1 + ex + static_cast<T>(1.f))) / nd4j::math::nd4j_pow<T>((ex + static_cast<T>(1.f)) , static_cast<T>(2.0f));
		}
	};


	template<typename T>
	class LogSigmoid {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_log(nd4j::math::nd4j_sigmoid<T>(d1));
		}
	};

	template<typename T>
	class LogSigmoidDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T ex = nd4j::math::nd4j_pow<T>(M_E, d1);
			return static_cast<T>(1.f) / (ex + static_cast<T>(1.f));
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

    template<typename T>
    class HardSigmoid {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return nd4j::math::nd4j_min<T>(static_cast<T>(1.0f), nd4j::math::nd4j_max<T>(static_cast<T>(0.0f), (static_cast<T>(0.2f)) * d1 + static_cast<T>(0.5f)));
        }
    };

    template<typename T>
    class HardSigmoidDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 < static_cast<T>(-2.5f) || d1 > static_cast<T>(2.5f) ? static_cast<T>(0.0f) : static_cast<T>(0.2f);
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
			if (min == static_cast<T>(0.0f) && max == static_cast<T>(1.0f)) {
				auto val = static_cast<T>(1.0f) / (static_cast<T>(1.0f) + nd4j::math::nd4j_exp<T>(-d1));
				return (nd4j::math::nd4j_floor<T>(val * (max - min)) + min);
			}

			auto ret = (nd4j::math::nd4j_floor<T>(d1 * (max - min)) + min);
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
	class Square {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * d1;
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
			return static_cast<T>(1.0f) / nd4j::math::nd4j_sqrt<T>(d1);
		}
	};

	template<typename T>
	class Rint {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_rint<T>(d1);
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
			return (d1 > static_cast<T>(0.0f)) - (d1 < static_cast<T>(0.0f));
		}
	};


	template<typename T>
	class TimesOneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return d1 * (static_cast<T>(1.0f) - d1);
		}
	};


	template<typename T>
	class RationalTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			// keep 2/3 as runtime variable, to match precision
			auto dis = (static_cast<T>(2.0f) / static_cast<T>(3.0f)) * d1;

			auto tanh = nd4j::math::nd4j_sgn<T>(dis) * (static_cast<T>(1.0f) - (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + nd4j::math::nd4j_abs<T>(dis) + nd4j::math::nd4j_pow<T>(dis, static_cast<T>(2.0f)) + static_cast<T>(1.41645f) * nd4j::math::nd4j_pow<T>(dis, static_cast<T>(4.0f)) )));
			return static_cast<T>(1.7159f) * tanh;
		}
	};

	template<typename T>
	class RationalTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			auto dis = (static_cast<T>(2.0f) / static_cast<T>(3.0f)) * d1;

			auto a = static_cast<T>(1.0f) + nd4j::math::nd4j_abs<T>(dis) + nd4j::math::nd4j_pow<T>(dis, static_cast<T>(2.)) + static_cast<T>(1.41645f) * nd4j::math::nd4j_pow<T>(dis, static_cast<T>(4.f));

			auto tDeriv = (static_cast<T>(1.0f) + nd4j::math::nd4j_sign<T>(dis) * (static_cast<T>(2.0f) * dis + static_cast<T>(4.0f) * static_cast<T>(1.41645f) * nd4j::math::nd4j_pow<T>(dis, static_cast<T>(3.f)))) / (a * a);

			return static_cast<T>(1.7159f) * (static_cast<T>(2.0f) / static_cast<T>(3.0f)) * tDeriv;
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
            return nd4j::math::nd4j_max<T>(static_cast<T>(0.0f), nd4j::math::nd4j_tanh<T>(d1));
        }
    };

    template<typename T>
    class RectifiedTanhDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 > static_cast<T>(0.0f) ? nd4j::math::nd4j_tanhderivative<T>(d1) : static_cast<T>(0.0f);
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
	class ASinh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_asinh<T>(d1);
		}
	};

	template<typename T>
	class ASinhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return static_cast<T>(1.f) / (nd4j::math::nd4j_sqrt(nd4j::math::nd4j_pow(d1, static_cast<T>(2.f)) + static_cast<T>(1.f)));
		}
	};

	template<typename T>
	class ACosh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_acosh<T>(d1);
		}
	};


	template<typename T>
	class ACoshDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return static_cast<T>(1.f) / (nd4j::math::nd4j_sqrt(d1 - static_cast<T>(1.f)) * nd4j::math::nd4j_sqrt(d1 + static_cast<T>(1.f)));
		}
	};



	template<typename T>
	class Ones {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return static_cast<T>(1.0f);
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
            return static_cast<T>(0.0f);
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

            auto mode = static_cast<int>(extraParams[2]);
            //nd4j_printf("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

			switch (mode) {
				case 0: // equals
					return nd4j::math::nd4j_abs<T>(d1 - compare) <= eps ? 1.0f : 0.0f;
				case 1: // not equals
					return nd4j::math::nd4j_abs<T>(d1 - compare) > eps ? 1.0f : 0.0f;
				case 2: // less_than
					return d1 < compare ? 1.0f : 0.0f;
				case 3: // greater_than
					return d1 > compare ? 1.0f : 0.0f;
				case 4: // less_or_equals_than
					return d1 <= compare ? 1.0f : 0.0f;
				case 5: // greater_or_equals_than
					return d1 >= compare ? 1.0f : 0.0f;
				case 6: // abs_less_than
					return nd4j::math::nd4j_abs<T>(d1) < compare ? 1.0f : 0.0f;
				case 7: // abs_greater_than
					return nd4j::math::nd4j_abs<T>(d1) > compare ? 1.0f : 0.0f;
				case 8: // is inf
					return nd4j::math::nd4j_isinf(d1) ? 1.0f : 0.0f;
				case 9: // is nan
					return nd4j::math::nd4j_isnan(d1) ? 1.0f : 0.0f;
				case 10:
					return (d1 == compare) ? 1.0f : 0.0f;
				case 11:
					return (d1 != compare) ? 1.0f : 0.0f;
				case 12: // abs_greater_or_equals_than
					return nd4j::math::nd4j_abs<T>(d1) >= compare ? 1.0f : 0.0f;
				case 13: // abs_less_or_equals_than
					return nd4j::math::nd4j_abs<T>(d1) <= compare ? 1.0f : 0.0f;
				default:
					printf("Undefined match condition: [%i]\n", mode);
			}

            return d1;
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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
	class RELU6 {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T relu = d1 < params[0] ? params[0] : d1;
			return relu < static_cast<T>(6.f) ? relu : static_cast<T>(6.f);
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
            return d1 > static_cast<T>(0.0f) ? static_cast<T>(SELU_LAMBDA) * d1 : static_cast<T>(SELU_LAMBDA) * (static_cast<T>(SELU_ALPHA) * nd4j::math::nd4j_exp<T>(d1) - static_cast<T>(SELU_ALPHA));
        }
    };

    template<typename T>
    class SELUDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static T op(T d1, T *params) {
            return d1 > static_cast<T>(0.0f) ? static_cast<T>(SELU_LAMBDA) : static_cast<T>(SELU_ALPHA) * static_cast<T>(SELU_LAMBDA) * nd4j::math::nd4j_exp<T>(d1);
        }
    };

	template<typename T>
	class LeakyRELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			if (d1 >= static_cast<T>(0.0f))
				return static_cast<T>(1.0f);
			else
				return params[0];
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
	class SinhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_cosh<T>(d1);
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
            return  static_cast<T>(1.0f) / nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_cos<T>(d1), static_cast<T>(2.0f));
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

		op_def static T op(T d1, T d2) {
			return nd4j::math::nd4j_atan2<T>(d2, d1);
		}

        op_def static T op(T d1, T d2, T *params) {
            return nd4j::math::nd4j_atan2<T>(d2, d1);
        }

		// op for MetaOps
		op_def static T op(T d1, T *params) {
			return nd4j::math::nd4j_atan2<T>(params[0], d1);
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
			T k = params[0];
			if (d1 * k > static_cast<T>(- MIN_CUTFOFF))
				return static_cast<T>(- MIN_CUTFOFF) / k;
			else if (d1 * k < static_cast<T>(MIN_CUTFOFF))
				return static_cast<T>(MIN_CUTFOFF) / k;
			return d1;
		}
	};



	template<typename T>
	class Step {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return (d1 > params[0] ? static_cast<T>(1.0f) : static_cast<T>(0.0f));
		}
	};



	template<typename T>
	class OneMinus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			return static_cast<T>(1.0f) - d1;
		}
	};

	template<typename T>
	class Sum {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction;
		}
	};




    template<typename T>
    class ShannonEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return nd4j::math::nd4j_pow<T>(d1, static_cast<T>(2.0f)) * nd4j::math::nd4j_log<T>(nd4j::math::nd4j_pow<T>(d1, static_cast<T>(2.0f)));
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return -reduction;
        }
    };


    template<typename T>
    class LogEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			//entropy is -sum(p(x) * log(p(x))); log entropy is log of this
			return nd4j::math::nd4j_log<T>(-reduction);
        }
    };

    template<typename T>
    class Entropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return -reduction;		//entropy is -sum(p(x) * log(p(x)))
        }
    };


    template<typename T>
    class ASum {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(reduction);
        }
    };


    template<typename T>
    class CountNonZero {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return d1 == static_cast<T>(0.0f) ? static_cast<T>(0.0f) : static_cast<T>(1.0f);
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
        }
    };


    template<typename T>
    class CountZero {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
        }

        op_def static T merge(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T update(T old, T opOutput, T *extraParams) {
            return opOutput + old;
        }

        op_def static T op(T d1, T *extraParams) {
            return d1 == static_cast<T>(0.0f) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
        }
    };

	template<typename T>
	class Prod {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(1.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Any {
	public:
		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction > static_cast<T>(0.0f) ? static_cast<T>(1.0f) : static_cast<T>(0.0f) ;
		}
	};


    template<typename T>
    class All {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(1.0f);
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction > static_cast<T>(0.0f) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
        }
    };

	template<typename T>
	class Mean {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction / (int) n;
		}
	};


    template<typename T>
    class AMean {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static T startingValue(const T *input) {
            return static_cast<T>(0.0f);
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return nd4j::math::nd4j_abs<T>(reduction) / static_cast<T>(n);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return reduction;
        }
    };


    template<typename T>
	class Norm1 {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction;
		}
	};


	template<typename T>
	class Norm2 {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return nd4j::math::nd4j_sqrt<T>(reduction);
		}

        op_def static T op(T d1, T *extraParams) {
            return d1 * d1;
        }
    };

	template<typename T>
	class SquaredNorm {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return reduction;
		}
	};

	template<typename T>
	class NormFrobenius {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T op(T d1, T *extraParams) {
			T v = nd4j::math::nd4j_abs(d1);
			return v * v;
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return nd4j::math::nd4j_sqrt<T>(reduction);
		}
	};

	template<typename T>
	class NormP {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
		}

		op_def static T merge(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}


		op_def static T update(T old, T opOutput, T *extraParams) {
			return opOutput + old;
		}

		op_def static T op(T d1, T *extraParams) {
			return nd4j::math::nd4j_pow(nd4j::math::nd4j_abs(d1), extraParams[0]);
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			return nd4j::math::nd4j_pow(reduction, static_cast<T>(1.0f) / extraParams[0]);
		}
	};

	template<typename T>
	class NormMax {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static T startingValue(const T *input) {
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
			// T bias = extraParams[1];
			// return (reduction - (nd4j::math::nd4j_pow<T>(bias, static_cast<T>(2.0f)) / static_cast<T>(n))) / (n - 1)
			return reduction / static_cast<T>(n - 1);
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
			return static_cast<T>(0.0f);
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

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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
			return static_cast<T>(0.0f);
		}

		op_def static  T postProcess(T reduction, Nd4jLong n, T *extraParams) {
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
		static _CUDA_D inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],static_cast<T>(d1 * d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],static_cast<T>(d2 * d2));

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
            return static_cast<T>(0.0f);
        }

        op_def static  T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            // num / denom
            return (static_cast<T>(1.0f)) - (extraParams[0] / extraParams[1]);
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
            return static_cast<T>(0.0f);
        }

        op_def static void aggregateExtraParams(T *extraParamsTotal, T *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
        __device__
		static inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],num(d1, d2));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], denom(d1, d2));

			return static_cast<T>(0.0f);
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
            return static_cast<T>(0.0f);
        }

        op_def static  T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return static_cast<T>(reduction / n);
        }

        op_def static T op(T d1, T d2, T *extraParams) {
            return (d1 == d2) ? static_cast<T>(0.0f) :  static_cast<T>(1.0f);
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
            return static_cast<T>(0.0f);
        }

        op_def static  T postProcess(T reduction, Nd4jLong n, T *extraParams) {
            return (static_cast<T>(1.0f)) - (reduction / (nd4j::math::nd4j_sqrt<T>(extraParams[0]) * nd4j::math::nd4j_sqrt<T>(extraParams[1])));
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
	static _CUDA_D inline T opAtomic(T d1, T d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0], nd4j::math::nd4j_abs<T>(d1) * nd4j::math::nd4j_abs<T>(d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], nd4j::math::nd4j_abs<T>(d2) * nd4j::math::nd4j_abs<T>(d2));

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
			return static_cast<T>(0.0f);
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParamsRef) {
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
            return static_cast<T>(0.0f);
        }

        op_def static T postProcess(T reduction, Nd4jLong n, T *extraParamsRef) {
            return reduction;
        }

        op_def static T op(T d1, T d2, T *extraParamsRef) {
        	
        	T eps = extraParamsRef[2];
    	    T diff = nd4j::math::nd4j_abs<T>(d1 - d2);
    	
    		// works well except in the range of very large numbers
    		if (diff <= eps)
    	    	return static_cast<T>(0.f);

    	    // Knuth approach
    	    // works well except in the range of very small numbers
		    if (diff <= nd4j::math::nd4j_max(nd4j::math::nd4j_abs(d1), nd4j::math::nd4j_abs(d2)) * eps)
		    	return static_cast<T>(0.f);
        
        	return static_cast<T>(1.f);
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
			return static_cast<T>(0.0f);
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParamsRef) {
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
			return static_cast<T>(0.0f);
		}

		op_def static T postProcess(T reduction, Nd4jLong n, T *extraParamsRef) {
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

            if (res == static_cast<T>(0.0f))
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
            return - nd4j::DataTypeUtils::max<T>();
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

            if (res == static_cast<T>(0.0f))
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
            return -nd4j::DataTypeUtils::max<T>();
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
			return -nd4j::DataTypeUtils::max<T>();
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
			return nd4j::DataTypeUtils::max<T>();
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
			return nd4j::DataTypeUtils::max<T>();
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

        static _CUDA_HD inline T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < static_cast<T>(0.0f))
					return val.variance();
				return ret;
			}
			return val.variance();
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

	template<typename T>
	class SummaryStatsStandardDeviation {
	public:

        static _CUDA_HD inline T getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<T> val) {
			if (biasCorrected) {
				T ret = val.varianceBiasCorrected();
				if (ret < static_cast<T>(0.0f))
					return nd4j::math::nd4j_sqrt(val.variance());
				else
					return nd4j::math::nd4j_sqrt(ret);
			}
			return  nd4j::math::nd4j_sqrt(val.variance());
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,T *extraParams) {
			return d1;
		}
	};

template<typename T>
	class DropOut {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		inline _CUDA_D static T op(T d1, T *params) {
			T prob = params[0];

#ifdef __CUDACC__
			T length = params[1];
            T tid = gridDim.x * blockDim.x + threadIdx.x;
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>(static_cast<T>(clock64()) * static_cast<T>(tid) + static_cast<T>(length) * static_cast<T>(tid)));
#else
			T rnd = static_cast<T>(rand() / RAND_MAX);
#endif
			return rnd >= prob ? static_cast<T>(0.0f) : d1;
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
            T rnd = nd4j::math::nd4j_abs<T>(nd4j::math::nd4j_cos<T>(static_cast<T>(clock64()) * static_cast<T>(tid) + static_cast<T>(length) * static_cast<T>(tid)));
#else
			T rnd = static_cast<T>(rand() / RAND_MAX);
#endif
			return rnd >= prob ? static_cast<T>(0.0f) : d1 / prob;
		}
	};


	template<typename T>
	class ReplaceNans {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static T op(T d1, T *params) {
			T replacement = params[0];
			return nd4j::math::nd4j_isnan(d1) ? replacement : d1 ;
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
                if (nd4j::math::nd4j_isinf(d1))
                    return d2;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (nd4j::math::nd4j_isnan(d1))
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
                if (nd4j::math::nd4j_isinf(d1))
                    return set;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (nd4j::math::nd4j_isnan(d1))
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
                if (nd4j::math::nd4j_isinf(d2))
                    return d2;
                else
                    return d1;
            else if (mode == 9) // is nan
                if (nd4j::math::nd4j_isnan(d2))
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


}

#endif
	
