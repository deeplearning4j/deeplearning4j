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
#define DOUBLE_PI_X X(2.0 * 3.14159265358979323846)

#define no_op_exec_special_bool 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_same 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, X *result, Nd4jLong *resultShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, Z *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation 	static const bool requiresSpecialAccumulation = false; static void execSpecial(X *x, Nd4jLong *xShapeInfo, Z *extraParams, Z *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
#define no_op_exec_special_accumulation_same 	static const bool requiresSpecialAccumulation = false; static void execSpecial(X *x, Nd4jLong *xShapeInfo, X *extraParams, X *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
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

#pragma omp declare reduction(sumT : float,double,float16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = omp_in + omp_out)\
                initializer (omp_priv=0)
#endif


namespace functions {
	namespace indexreduce {
		template <typename T>
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
	template <typename X, typename Y, typename Z>
	class Add {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 + d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d1 + d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return d1 + params[0];
		}

		op_def static Z startingValue() {
			return static_cast<Z>(0.f);
		}
	};

	template <typename X, typename Y>
	class NewAdd {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return d1 + d2;
        }
	};

	template <typename X, typename Y, typename Z>
	class Subtract {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 - d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d1 - d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return d1 - params[0];
		}

	};

	template <typename X, typename Y, typename Z>
	class SquaredSubtract {
	public:
		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_pow<X, Y, Z>(d1 - d2, static_cast<Y>(2.f));
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_pow<X, Y, Z>(d1 - d2, static_cast<Y>(2.f));
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return nd4j::math::nd4j_pow<X, Y, Z>(d1 - params[0], static_cast<X>(2.f));
		}
	};

	template <typename X, typename Y, typename Z>
	class ReverseSubtract {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 - d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d2 - d1;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return params[0] - d1;
		}
	};


	template <typename X, typename Y, typename Z>
	class LogPoisonLossFull {

	public:
		op_def static Z op(X z, Y c) {
			return (nd4j::math::nd4j_exp<X, Z>(c) - z * c  + (z * nd4j::math::nd4j_log<X, Z>(z) - z + static_cast<X>(0.5f) * nd4j::math::nd4j_log<X, Z>(DOUBLE_PI_X * z)));
		}

		op_def static Z op(X z, Y c, Z *params) {
			return (nd4j::math::nd4j_exp<X, Z>(c) - z * c  + (z * nd4j::math::nd4j_log<X, Z>(z) - z + static_cast<X>(0.5f) * nd4j::math::nd4j_log<X, Z>(DOUBLE_PI_X * z)));
		}

		op_def static Z op(X z) {
			return (z * nd4j::math::nd4j_log<X, Z>(z) - z + static_cast<X>(0.5f) * nd4j::math::nd4j_log<X, Z>(DOUBLE_PI_X * z));
		}

		// op for MetaOps
		op_def static X op(X z, Y *params) {
			return (nd4j::math::nd4j_exp<X, X>(params[0]) - z * params[0]  + (z * nd4j::math::nd4j_log<X, Z>(z) - z + static_cast<X>(0.5f) * nd4j::math::nd4j_log<X, Z>(DOUBLE_PI_X * z)));
		}
	};

	template <typename X, typename Y, typename Z>
	class LogPoisonLoss {

	public:
		op_def static Z op(X z, Y c) {
			return (nd4j::math::nd4j_exp<X, Z>(c) - z * c);
		}

		op_def static Z op(X z, Y c, Z *params) {
			return (nd4j::math::nd4j_exp<X, Z>(c) - z * c);
		}

		op_def static Z op(X z) {
			return z;
		}

		// op for MetaOps
		op_def static Z op(X z, Y *params) {
			return (nd4j::math::nd4j_exp<X, Z>(params[0]) - z * params[0]);
		}
	};

	template <typename X, typename Y, typename Z>
	class Multiply {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 * d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d1 * d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return d1 * params[0];
		}

		op_def static Z startingValue() {
			return static_cast<Z>(1.f);
		}
	};

	template <typename X, typename Y, typename Z>
	class Divide {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 / d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d1 / d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return d1 / params[0];
		}

		op_def static Z startingValue() {
			return static_cast<Z>(1);
		}
	};

	template <typename X, typename Y, typename Z>
	class SafeDivide {
	public:
		op_def static Z op(X d1, Y d2) {
			if(d2 == static_cast<Y>(0))
				return static_cast<Z>(0);
			return d1 / d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			if(d2 == static_cast<Y>(0))
				return static_cast<X>(0);
			return d1 / d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			if(params[0] == static_cast<Y>(0))
				return static_cast<Z>(0);
			return d1 / params[0];
		}
	};

    template <typename X, typename Y, typename Z>
    class FloorDiv {
    public:
        op_def static Z op(X d1, Y d2) {
            return nd4j::math::nd4j_floor<X,Z>(d1 / d2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            return nd4j::math::nd4j_floor<X,Z>(d1 / d2);
        }

        op_def static Z op(X d1) {
            return nd4j::math::nd4j_floor<X,Z>(d1);
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            return nd4j::math::nd4j_floor<X,Z>(d1 / params[0]);
        }
    };

    template <typename X, typename Y, typename Z>
    class TruncateDiv {
    public:
        op_def static Z op(X d1, Y d2) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<Z>(i1 / i2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<Z>(i1 / i2);
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(params[0]);
            return static_cast<Z>(i1 / i2);
        }
    };

    template <typename X, typename Y, typename Z>
    class Remainder {
    public:
        op_def static Z op(X d1, Y d2) {
            return nd4j::math::nd4j_remainder<X, Y, Z>(d1, d2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            return nd4j::math::nd4j_remainder<X, Y, Z>(d1, d2);
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            return nd4j::math::nd4j_remainder<X, Y, Z>(d1, params[0]);
        }
    };

    template <typename X, typename Y, typename Z>
    class FMod {
    public:
        op_def static Z op(X d1, Y d2) {
            return nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            return nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            return nd4j::math::nd4j_fmod<X, Y, Z>(d1, params[0]);
        }
    };

	template <typename X, typename Y, typename Z>
    class FloorMod {
    public:
        op_def static Z op(X d1, Y d2) {
			auto m = nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);;
            return (d1 < static_cast<X>(0)) == (d2 < static_cast<Y>(0)) ? m : nd4j::math::nd4j_fmod<X, Y, Z>(m + d2, d2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto m = nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);
			return (d1 < static_cast<X>(0.0f)) == (d2 < static_cast<Y>(0)) ? m : nd4j::math::nd4j_fmod<X, Y, Z>(m + d2, d2);
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            return op(d1, params[0]);
        }
    };

	template <typename X, typename Y, typename Z>
	class ReverseDivide {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 / d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d2 / d1;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return params[0] / d1;
		}
	};

	template <typename X, typename Y, typename Z>
	class Copy {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return params[0];
		}
	};

	template <typename X, typename Y, typename Z>
	class Copy2 {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(params[0]);
		}
	};

	template <typename X, typename Y, typename Z>
	class Axpy {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			auto alpha = params[0];
			return alpha * d1 + d2;
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};

	template <typename X, typename Y, typename Z>
	class And {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			auto comp = params[0];
			return d1 != comp && static_cast<X>(d2) != comp ? static_cast<X>(1) : static_cast<X>(0);
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, X *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Y, typename Z>
	class Or {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			auto comp = params[0];

			return d1 != comp || static_cast<X>(d2) != comp ? static_cast<Z>(1) : static_cast<Z>(0);
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Y, typename Z>
	class Xor {
	public:
		op_def static Z op(X d1, Y d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			auto comp = params[0];

			return ((d1 == comp && static_cast<X>(d2) != comp)||(d1 != comp && static_cast<X>(d2) == comp)) ? static_cast<X>(1) : static_cast<X>(0);
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};

	template <typename X, typename Z>
	class Not {
	public:
		no_op_exec_special_bool
		no_op_exec_special_cuda

		op_def static Z op(X d1, X *params) {
			auto comp = params[0];

			return d1 == comp ? static_cast<Z>(1) : static_cast<Z>(0);
		}
	};

	template <typename X, typename Y, typename Z>
	class LogicalNot {
	public:
		op_def static Z op(X d1, Y d2) {
			return !((int) d1  && (int) d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<X>(!(static_cast<int>(d1)  && static_cast<int>(d2)));
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<X>(119);
		}
	};

	template <typename X, typename Y, typename Z>
	class LogicalXor {
	public:
		op_def static Z op(X d1, Y d2) {
		    auto i1 = static_cast<int>(d1);
		    auto i2 = static_cast<int>(d2);

			return  (i1 | i2) &~ (i1 & i2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Y, typename Z>
	class LogicalAnd {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<int>(d1)  & static_cast<int>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(Y d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Y, typename Z>
	class LogicalOr {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<int>(d1) | static_cast<int>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
            return op(d1, d2);
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<X>(119);
		}
	};


	template <typename X, typename Y, typename Z>
	class Mod {
	public:
		/*

		 // just a optional note, feel free to remove later

		op_def static half op(half d1, half d2, half *params) {
			return __float2half(simdOps::Mod<float>::op(__half2float(d1), __half2float(d2), nullptr));
		}
		 */

		op_def static Z op(X d1, Y d2) {
            return static_cast<int>(d1) % static_cast<int>(d2);
        }

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		// op for MetaOp
		op_def static Z op(X d1, Y *params) {
			return op(d1, params[0]);
		}
	};

	template <typename X, typename Y, typename Z>
	class ReverseMod {
	public:
        op_def static Z op(X d1, Y d2) {
            return static_cast<int>(d2) % static_cast<int>(d1);
        }

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		// op for MetaOp
		op_def static Z op(X d1, Y *params) {
			return op(d1, params[0]);
		}
	};

	/**
	* Whether 2 elements in an array
	* are epsilion equal
	*/
	template <typename X, typename Y, typename Z>
	class Epsilon {
	public:
		op_def static Z op(X d1, Y d2, Z *params) {
			X diff = d1 - d2;
			X absDiff = nd4j::math::nd4j_abs<X>(diff);
			if (absDiff <= static_cast<X>(MIN))
				return static_cast<Z>(1);
			return static_cast<Z>(0);
		}

		op_def static Z op(X d1, Y *params) {
			return d1;
		}
	};


	template <typename X, typename Y, typename Z>
	class EqualTo {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 == static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X, typename Y, typename Z>
	class NotEqualTo {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 != static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};



	template <typename X, typename Y, typename Z>
	class GreaterThanOrEqual {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 >= static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X, typename Y, typename Z>
	class GreaterThan {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 > static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X, typename Y, typename Z>
	class LessThan {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 < static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X, typename Y, typename Z>
	class LessThanOrEqual {
	public:
		op_def static Z op(X d1, Y d2) {
			return d1 <= static_cast<X>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X>
	class Abs {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_abs<X>(d1);
		}
	};


	template <typename X>
	class Ceiling {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_ceil<X,X>(d1);
		}
	};


	template <typename X, typename Z>
	class Cosine {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_cos<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class Exp {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_exp<X, Z>(d1);
		}
	};


	template <typename X, typename Z>
	class HardTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return ((d1 >= static_cast<X>(-1) && d1 <= static_cast<X>(1)) ? static_cast<Z>(1) : static_cast<Z>(0));
		}
	};


	template <typename X, typename Z>
	class HardTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			if (d1 < static_cast<X>(-1))
				return static_cast<Z>(-1);
			else if (d1 > static_cast<X>(1))
				return static_cast<Z>(1);
			else
				return d1;

		}
	};


	template <typename X>
	class Floor {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_floor<X,X>(d1);
		}
	};


	template <typename X, typename Z>
	class Log {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_log<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class Log1p {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_log<X, Z>(1 + d1);
		}
	};

	template <typename X, typename Y, typename Z>
	class LogX {
	public:

		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_log<X, Z>(d1) / nd4j::math::nd4j_log<Y, Z>(d2) ;
		}
	};

    template <typename X>
    class StabilizeFP16 {
    public:
        no_op_exec_special_same
        no_op_exec_special_cuda

        op_def static X op(X d1, X *params) {
            if (d1 <= static_cast<X>(0))
            	return static_cast<X>(nd4j::DataTypeUtils::min<float16>());
            else return d1;
        }
    };

    template <typename X, typename Z>
    class StabilizeX {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            if (d1 <= static_cast<X>(0))
            	return nd4j::DataTypeUtils::min<Z>();
            else return d1;
        }
    };

	template <typename X, typename Z>
	class SpecialDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return d1 * (static_cast<X>(1) - d1);
		}
	};


	template <typename X>
	class Neg {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return -d1;
		}
	};

	template <typename X, typename Z>
	class Erf {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_erf<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class Erfc {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_erfc<X,Z>(d1);
		}
	};

	template <typename X>
	class Reciprocal {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda
//		op_def static T op(T d1) {
//			return (T(1.0f) / d1);
//		}
		// op for MetaOps
		op_def static X op(X d1, X *params) {
			return (static_cast<X>(1) / d1);
		}
	};

	template <typename X, typename Z>
	class Sqr {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_pow<X, X, Z>(d1, static_cast<X>(2));
		}

		op_def static Z op(X d1) {
			return nd4j::math::nd4j_pow<X, X, Z>(d1, static_cast<X>(2));
		}
	};


	template <typename X, typename Y, typename Z>
	class RelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_re<X>(d1, d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static X op(X d1) {
			return static_cast<Z>(0);
		}
	};

	template <typename X, typename Y, typename Z>
	class BinaryRelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			X threshold = params[0];
			return nd4j::math::nd4j_re<X>(d1, d2) > threshold ? static_cast<X>(1) : static_cast<X>(0);
		}

		op_def static X op(X d1) {
			return static_cast<X>(0);
		}
	};

	template <typename X, typename Y, typename Z>
	class BinaryMinimumAbsoluteRelativeError {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, X *params) {
			X d2 = params[0];
			X thresholdRelative = params[1];
			X thresholdAbsolute = params[2];
			return nd4j::math::nd4j_re<X>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<X>(d1 - d2) < thresholdAbsolute ? static_cast<X>(0) : static_cast<X>(1)) : static_cast<X>(0);
 		}

		op_def static Z op(X d1, Y d2, Z *params) {
			X thresholdRelative = params[0];
			X thresholdAbsolute = params[1];
			return nd4j::math::nd4j_re<X>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<X>(d1 - d2) < thresholdAbsolute ? static_cast<X>(0) : static_cast<X>(1)) : static_cast<X>(0);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(0);
		}
	};



	template <typename X, typename Y, typename Z>
	class Pow {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_pow<X, X, Z>(d1, params[0]);
		}

		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_pow<X, Y, Z>(d1, d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_pow<X, Y, Z>(d1, d2);
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};


	template <typename X, typename Y, typename Z>
	class PowDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return params[0] * nd4j::math::nd4j_pow<X, X, Z>(d1, params[0] - static_cast<X>(1));
		}

		op_def static Z op(X d1, Y d2) {
			return d2 * nd4j::math::nd4j_pow<X, Y, Z>(d1, d2 - static_cast<Y>(1));
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return d2 * nd4j::math::nd4j_pow<X, Y, X>(d1, d2 - static_cast<Y>(1));
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};


	template <typename X>
	class Round {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_round<X,X>(d1);
		}
	};

	template <typename X, typename Z>
	class IsNan {
	public:
		no_op_exec_special_bool
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isnan(d1) ? static_cast<X>(1) : static_cast<X>(0);
		}

        op_def static Z startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }


        op_def static Z update(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }


        op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
	};


	template <typename X, typename Z>
	class Expm1 {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_exp<X, Z>(d1) - static_cast<Z>(1);
		}
	};

	template <typename X, typename Z>
	class IsInf {
	public:
		no_op_exec_special_bool
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isinf<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
		}

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0);
        }

        op_def static Z merge(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }


        op_def static Z update(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }


        op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
	};


	template <typename X, typename Z>
	class IsInfOrNan{
	public:
		no_op_exec_special_bool
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isfin<X>(d1) ? static_cast<Z>(0) : static_cast<Z>(1);
		}

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}


		op_def static X update(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction;
		}
	};



	template <typename X, typename Z>
	class IsFinite {
	public:
		no_op_exec_special_bool
		no_op_exec_special_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isfin<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
		}

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0);
        }

        op_def static Z merge(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }


        op_def static Z update(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
	};


	template <typename X>
	class ClipByValue {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			if (d1 > params[1])
				return params[1];
			else if (d1 < params[0])
				return params[0];
			else return d1;
		}
	};

	template <typename X, typename Y, typename Z>
	class LstmClip {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			X _v = (X) d2;
			if (d1 > _v)
				return _v;
			else if (d1 < _v)
				return _v;
			else return d1;
		}
	};

	template <typename X, typename Z>
	class Swish {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return d1 * nd4j::math::nd4j_sigmoid<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class SwishDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			Z ex = nd4j::math::nd4j_pow<X, X, Z>(static_cast<X>(M_E), d1);
			return (ex * (d1 + ex + static_cast<Z>(1))) / nd4j::math::nd4j_pow<Z, X, Z>((ex + static_cast<Z>(1)) , static_cast<X>(2));
		}
	};


	template <typename X, typename Z>
	class LogSigmoid {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_log<Z, Z>(nd4j::math::nd4j_sigmoid<X, Z>(d1));
		}
	};

	template <typename X, typename Z>
	class LogSigmoidDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			Z ex = nd4j::math::nd4j_pow<X, X, Z>(M_E, d1);
			return static_cast<Z>(1) / (ex + static_cast<Z>(1));
		}
	};

	template <typename X, typename Z>
	class Sigmoid {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_sigmoid<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class SigmoidDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_sigmoidderivative<X, Z>(d1);
		}
	};

	template <typename X, typename Y, typename Z>
	class SigmoidDerivativeE {
	public:
		op_def static Z op(X d1, Y d2, X *params) {
            Z s = nd4j::math::nd4j_sigmoid<X, Z>(d1);
            return (Z) d2 * (s * ((Z) 1.0f - s));
		}
	};

    template <typename X, typename Y, typename Z>
    class SoftplusDerivativeE {
    public:
        op_def static Z op(X d1, Y d2, X *params) {
            Z p = nd4j::math::nd4j_pow<X, X, Z>(static_cast<X>(M_E), d1);
            return (Z) d2 * (p / (p + 1.f));
        }
    };

    template <typename X, typename Y, typename Z>
    class SoftsignDerivativeE {
    public:
        op_def static Z op(X d1, Y d2, X *params) {
            X f = (X) 1.0f + nd4j::math::nd4j_abs<X>(d1);
            return (X) d2 * ((X) 1.0f / (f * f));
        }
    };

    template <typename X, typename Y, typename Z>
    class TanhDerivativeE {
    public:
        op_def static Z op(X d1, Y d2, X *params) {
            Z t = nd4j::math::nd4j_tanh<X,Z>(d1);
            return (Z) d2 * ((Z) 1.0f - (t * t));
        }
    };

    template <typename X, typename Z>
    class HardSigmoid {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return nd4j::math::nd4j_min<X>(static_cast<X>(1), nd4j::math::nd4j_max<X>(static_cast<X>(0), (static_cast<X>(0.2f)) * d1 + static_cast<X>(0.5f)));
        }
    };

    template <typename X, typename Z>
    class HardSigmoidDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return d1 < static_cast<X>(-2.5f) || d1 > static_cast<X>(2.5f) ? static_cast<Z>(0) : static_cast<Z>(0.2f);
        }
    };


	/**
	* Scale to be between a min and max
	*/
	template <typename X, typename Z>
	class SetRange {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			auto min = params[0];
			auto max = params[1];
			if (d1 >= min && d1 <= max)
				return d1;
			if (min == static_cast<X>(0) && max == static_cast<X>(1)) {
				auto val = static_cast<X>(1) / (static_cast<X>(1) + nd4j::math::nd4j_exp<X, X>(-d1));
				return (nd4j::math::nd4j_floor<X,Z>(val * (max - min)) + min);
			}

			return (nd4j::math::nd4j_floor<X,Z>(d1 * (max - min)) + min);
		}
	};


	template <typename X, typename Z>
	class Sin {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_sin<X,Z>(d1);
		}
	};

	template <typename X>
	class Square {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return d1 * d1;
		}
	};

	template <typename X, typename Z>
	class Sqrt {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_sqrt<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class RSqrt {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return static_cast<Z>(1) / nd4j::math::nd4j_sqrt<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class Rint {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_rint<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class SoftPlus {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::softplus<X, Z>(d1);
		}
	};


	template <typename X>
	class Sign {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return (d1 > static_cast<X>(0)) - (d1 < static_cast<X>(0));
		}
	};


	template <typename X>
	class TimesOneMinus {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return d1 * (static_cast<X>(1) - d1);
		}
	};


	template <typename X, typename Z>
	class RationalTanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			// keep 2/3 as runtime variable, to match precision
			auto dis = (static_cast<X>(2) / static_cast<X>(3)) * d1;

			auto tanh = nd4j::math::nd4j_sgn<X,Z>(dis) * (static_cast<Z>(1) - (static_cast<Z>(1) / (static_cast<Z>(1) + nd4j::math::nd4j_abs<X>(dis) + nd4j::math::nd4j_pow<X, X, Z>(dis, static_cast<X>(2)) + static_cast<Z>(1.41645f) * nd4j::math::nd4j_pow<X, X, Z>(dis, static_cast<X>(4)) )));
			return static_cast<Z>(1.7159f) * tanh;
		}
	};

	template <typename X, typename Z>
	class RationalTanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			auto dis = (static_cast<X>(2) / static_cast<X>(3)) * d1;

			auto a = static_cast<X>(1) + nd4j::math::nd4j_abs<X>(dis) + nd4j::math::nd4j_pow<X, X, Z>(dis, static_cast<X>(2)) + static_cast<Z>(1.41645f) * nd4j::math::nd4j_pow<X, X, Z>(dis, static_cast<X>(4));

			auto tDeriv = (static_cast<X>(1) + nd4j::math::nd4j_sign<X,X>(dis) * (static_cast<X>(2) * dis + static_cast<Z>(4) * static_cast<Z>(1.41645f) * nd4j::math::nd4j_pow<X, X, Z>(dis, static_cast<X>(3)))) / (a * a);

			return static_cast<Z>(1.7159f) * (static_cast<Z>(2) / static_cast<Z>(3)) * tDeriv;
		}
	};

	template <typename X, typename Z>
	class Tanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_tanh<X, Z>(d1);
		}
	};

    template <typename X, typename Z>
    class RectifiedTanh {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return nd4j::math::nd4j_max<Z>(static_cast<Z>(0), nd4j::math::nd4j_tanh<X,Z>(d1));
        }
    };

    template <typename X, typename Z>
    class RectifiedTanhDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return d1 > static_cast<X>(0) ? nd4j::math::nd4j_tanhderivative<X,Z>(d1) : static_cast<Z>(0);
        }
    };

	template <typename X, typename Z>
	class ATanh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_atanh<X,Z>(d1);
		}
	};

	template <typename X, typename Z>
	class TanhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_tanhderivative<X,Z>(d1);
		}
	};

	template <typename X>
	class Cube {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return d1 * d1 * d1;
		}
	};


	template <typename X>
	class CubeDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

			op_def static X op(X d1, X *params) {
			return 3 * d1 * d1;
		}
	};

	template <typename X, typename Z>
	class ACos {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_acos<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class ASinh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_asinh<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class ASinhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return static_cast<Z>(1) / (nd4j::math::nd4j_sqrt<X, Z>(nd4j::math::nd4j_pow<X, X, Z>(d1, static_cast<X>(2)) + static_cast<Z>(1)));
		}
	};

	template <typename X, typename Z>
	class ACosh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_acosh<X, Z>(d1);
		}
	};


	template <typename X, typename Z>
	class ACoshDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return static_cast<Z>(1.f) / (nd4j::math::nd4j_sqrt<X, Z>(d1 - static_cast<Z>(1.f)) * nd4j::math::nd4j_sqrt<X, Z>(d1 + static_cast<X>(1.f)));
		}
	};



	template <typename X>
	class Ones {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1.0f);
		}
	};



	template <typename X, typename Z>
	class SoftSign {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_softsign<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class SoftSignDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_softsignderivative<X,Z>(d1);
		}
	};

    template <typename X, typename Z>
    class MatchConditionBool {
    public:
        no_op_exec_special_bool
        no_op_exec_special_cuda

        // this op return 1.0 if condition met, 0.0 otherwise
        op_def static Z op(X d1, X *extraParams) {
            X compare = extraParams[0];
            X eps = extraParams[1];

            auto mode = static_cast<int>(extraParams[2]);
            //nd4j_printf("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

            switch (mode) {
                case 0: // equals
                    return nd4j::math::nd4j_abs<X>(d1 - compare) <= eps ? true : false;
                case 1: // not equals
                    return nd4j::math::nd4j_abs<X>(d1 - compare) > eps ? true : false;
                case 2: // less_than
                    return d1 < compare ? true : false;
                case 3: // greater_than
                    return d1 > compare ? true : false;
                case 4: // less_or_equals_than
                    return d1 <= compare ? true : false;
                case 5: // greater_or_equals_than
                    return d1 >= compare ? true : false;
                case 6: // abs_less_than
                    return nd4j::math::nd4j_abs<X>(d1) < compare ? true : false;
                case 7: // abs_greater_than
                    return nd4j::math::nd4j_abs<X>(d1) > compare ? true : false;
                case 8: // is inf
                    return nd4j::math::nd4j_isinf(d1) ? true : false;
                case 9: // is nan
                    return nd4j::math::nd4j_isnan(d1) ? true : false;
                case 10:
                    return (d1 == compare) ? true : false;
                case 11:
                    return (d1 != compare) ? true : false;
                case 12: // abs_greater_or_equals_than
                    return nd4j::math::nd4j_abs<X>(d1) >= compare ? true : false;
                case 13: // abs_less_or_equals_than
                    return nd4j::math::nd4j_abs<X>(d1) <= compare ? true : false;
                default:
                    printf("Undefined match condition: [%i]\n", mode);
            }

            return d1;
        }
    };

    template <typename X>
    class MatchCondition {
    public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return old + opOutput;
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return old + opOutput;
        }

        // this op return 1.0 if condition met, 0.0 otherwise
        op_def static X op(X d1, X *extraParams) {
            X compare = extraParams[0];
            X eps = extraParams[1];

            auto mode = static_cast<int>(extraParams[2]);
            //nd4j_printf("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

			switch (mode) {
				case 0: // equals
					return nd4j::math::nd4j_abs<X>(d1 - compare) <= eps ? 1.0f : 0.0f;
				case 1: // not equals
					return nd4j::math::nd4j_abs<X>(d1 - compare) > eps ? 1.0f : 0.0f;
				case 2: // less_than
					return d1 < compare ? 1.0f : 0.0f;
				case 3: // greater_than
					return d1 > compare ? 1.0f : 0.0f;
				case 4: // less_or_equals_than
					return d1 <= compare ? 1.0f : 0.0f;
				case 5: // greater_or_equals_than
					return d1 >= compare ? 1.0f : 0.0f;
				case 6: // abs_less_than
					return nd4j::math::nd4j_abs<X>(d1) < compare ? 1.0f : 0.0f;
				case 7: // abs_greater_than
					return nd4j::math::nd4j_abs<X>(d1) > compare ? 1.0f : 0.0f;
				case 8: // is inf
					return nd4j::math::nd4j_isinf(d1) ? 1.0f : 0.0f;
				case 9: // is nan
					return nd4j::math::nd4j_isnan(d1) ? 1.0f : 0.0f;
				case 10:
					return (d1 == compare) ? 1.0f : 0.0f;
				case 11:
					return (d1 != compare) ? 1.0f : 0.0f;
				case 12: // abs_greater_or_equals_than
					return nd4j::math::nd4j_abs<X>(d1) >= compare ? 1.0f : 0.0f;
				case 13: // abs_less_or_equals_than
					return nd4j::math::nd4j_abs<X>(d1) <= compare ? 1.0f : 0.0f;
				default:
					printf("Undefined match condition: [%i]\n", mode);
			}

            return d1;
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
    };

	template <typename X, typename Z>
	class ELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static X op(X d1, Z *params) {
			return nd4j::math::nd4j_elu<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class ELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_eluderivative<X,Z>(d1);
		}
	};


	template <typename X>
	class RELU {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
		    X t = params == nullptr ? static_cast<X>(0.0f) : params[0];
			return d1 < t ? t : d1;
		}
	};

    template <typename X, typename Y, typename Z>
    class SXELogitsSmoother {
    public:
        op_def static Z op(X d1, Y d2, Z *params) {
            return d1 * ((X)1.f - (X) d2) + (X)(0.5f) * (X) d2;
        }
    };

    template <typename X, typename Y>
    class SXELossWithLogits {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return nd4j::math::nd4j_max<X>(d1, (X)0.f) - d1 * (X) d2 + nd4j::math::nd4j_log<X, X>((X)1.f + nd4j::math::nd4j_exp<X, X>(-nd4j::math::nd4j_abs<X>(d1)));
        }
    };

	template <typename X, typename Y>
	class RELUDerivativeE {
	public:
		op_def static X op(X d1, Y d2, X *params) {
			X t = params == nullptr ? (X) 0.0f : params[0];
			return d1 > t ? (X) d2  : (X) 0.0f;
		}
	};

    template <typename X, typename Y>
    class RELU6DerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            if((X)0.f < d1 && d1 < (X)6.f)
                return (X) d2;                    // derivative = 1
            else
                return (X) 0.f;                   // derivative = 0
        }
    };

    template <typename X, typename Y>
    class LRELUDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
        	return d1 >= static_cast<X>(0.0f) ? static_cast<X>(d2) : static_cast<X>(0.0f);
        }
    };

	template <typename X, typename Y>
	class CUBEDerivativeE {
	public:

	    op_def static X op(X d1, Y d2, X *params) {
            return static_cast<X>(d2) * (3 * d1 * d1);
	    }
	};

    template <typename X, typename Y>
    class ReduceNorm1E {
    public:

        op_def static X op(X d1, Y d2, X *params) {
            X _d2 = static_cast<X>(d2);
            return (d1 >= X(0.f) ? _d2: -_d2);
        }
    };

    template <typename X, typename Y>
    class ELUDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return static_cast<X>(d2) * nd4j::math::nd4j_eluderivative<X>(d1);
        }
    };

    template <typename X, typename Y>
    class HardSigmoidDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return static_cast<X>(d2) * simdOps::HardSigmoidDerivative<X,Y>::op(d1, nullptr);
        }
    };

    template <typename X, typename Y>
    class HardTanhDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return static_cast<X>(d2) * simdOps::HardTanhDerivative<X,Y>::op(d1, nullptr);
        }
    };

    template <typename X, typename Y>
    class RationalTanhDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return static_cast<X>(d2) * simdOps::RationalTanhDerivative<X,Y>::op(d1, nullptr);
        }
    };

    template <typename X, typename Y>
    class RectifiedTanhDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return d1 > (X) 0.0f ? (X) d2 * (nd4j::math::nd4j_tanhderivative<X>(d1)) : (X) 0.0f;
        }
    };

	template <typename X>
	class RELU6 {
	public:
	    no_op_exec_special_same

		op_def static X op(X d1, X *params) {
			X relu = d1 < params[0] ? params[0] : d1;
			return relu < static_cast<X>(6) ? relu : static_cast<X>(6);
		}
	};

	template <typename X, typename Y, typename Z>
	class LeakyRELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_leakyrelu<X,Z>(d1, d2);
		}
	};

    template <typename X, typename Z>
    class SELU {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return d1 > static_cast<X>(0.0f) ? static_cast<Z>(SELU_LAMBDA) * static_cast<Z>(d1) : static_cast<Z>(SELU_LAMBDA) * (static_cast<Z>(SELU_ALPHA) * nd4j::math::nd4j_exp<X, Z>(d1) - static_cast<Z>(SELU_ALPHA));
        }
    };

    template <typename X, typename Z>
    class SELUDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return d1 > static_cast<X>(0) ? static_cast<Z>(SELU_LAMBDA) : static_cast<Z>(SELU_ALPHA) * static_cast<Z>(SELU_LAMBDA) * nd4j::math::nd4j_exp<X, Z>(d1);
        }
    };

    template <typename X, typename Y>
    class SELUDerivativeE {
    public:
        op_def static X op(X d1, Y d2, X *params) {
            return (Y) d2 * simdOps::SELUDerivative<X, Y>::op(d1, nullptr);
        }
    };

	template <typename X, typename Y, typename Z>
	class LeakyRELUDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			if (d1 >= static_cast<X>(0))
				return static_cast<Z>(1);
			else
				return static_cast<Z>(d2);
		}
	};


	template <typename X, typename Z>
	class ASin {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_asin<X,Z>(d1);
		}
	};

	template <typename X, typename Z>
	class Sinh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_sinh<X,Z>(d1);
		}
	};

	template <typename X, typename Z>
	class SinhDerivative {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_cosh<X, Z>(d1);
		}
	};

	template <typename X, typename Z>
	class Cosh {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_cosh<X,Z>(d1);
		}
	};


	template <typename X, typename Z>
	class Tan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_tan<X,Z>(d1);
		}
	};

    template <typename X, typename Z>
    class TanDerivative {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return  static_cast<Z>(1) / nd4j::math::nd4j_pow<Z, Z, Z>(nd4j::math::nd4j_cos<X, Z>(d1), static_cast<Z>(2.0f));
        }
    };

	template <typename X, typename Z>
	class ATan {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Z *params) {
			return nd4j::math::nd4j_atan<X, Z>(d1);
		}
	};

    template <typename X, typename Y, typename Z>
    class Atan2 {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_atan2<X, Z>(d2, d1);
		}

        op_def static Z op(X d1, Y d2, Z *params) {
            return op(d1, d2);
        }

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return op(d1, params[0]);
		}
    };


	template <typename X>
	class Identity {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X>
	class Stabilize {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			X k = params[0];
			if (d1 * k > static_cast<X>(- MIN_CUTFOFF))
				return static_cast<X>(- MIN_CUTFOFF) / k;
			else if (d1 * k < static_cast<X>(MIN_CUTFOFF))
				return static_cast<X>(MIN_CUTFOFF) / k;
			return d1;
		}
	};



	template <typename X>
	class Step {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return (d1 > params[0] ? static_cast<X>(1) : static_cast<X>(0));
		}
	};



	template <typename X>
	class OneMinus {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1) - d1;
		}
	};

	template <typename X>
	class Sum {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static X update(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, X *extraParams) {
			return d1;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction;
		}
	};




    template <typename X, typename Z>
    class ShannonEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X update(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X op(X d1, Z *extraParams) {
            return nd4j::math::nd4j_pow<X, X, X>(d1, static_cast<X>(2)) * nd4j::math::nd4j_log<X, X>(nd4j::math::nd4j_pow<X, X, X>(d1, static_cast<X>(2.0f)));
        }

        op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
            return -reduction;
        }
    };


    template <typename X, typename Z>
    class LogEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X update(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X op(X d1, Z *extraParams) {
			return d1 * nd4j::math::nd4j_log<X, X>(d1);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			//entropy is -sum(p(x) * log(p(x))); log entropy is log of this
			return nd4j::math::nd4j_log<X, X>(-reduction);
        }
    };

    template <typename X, typename Z>
    class Entropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X update(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X op(X d1, Z *extraParams) {
            return d1 * nd4j::math::nd4j_log<X, X>(d1);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
            return -reduction;		//entropy is -sum(p(x) * log(p(x)))
        }
    };


    template <typename X>
    class ASum {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(opOutput) + nd4j::math::nd4j_abs<X>(old);
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(opOutput) + nd4j::math::nd4j_abs<X>(old);
        }

        op_def static X op(X d1, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(d1);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(reduction);
        }
    };


    template <typename X, typename Z>
    class CountNonZero {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static X op(X d1, X *extraParams) {
            return d1 == static_cast<X>(0.0f) ? static_cast<X>(0.0f) : static_cast<X>(1.0f);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
    };


    template <typename X, typename Z>
    class CountZero {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static X merge(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X update(X old, X opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static X op(X d1, Z *extraParams) {
            return d1 == static_cast<X>(0) ? static_cast<X>(1) : static_cast<X>(0);
        }

        op_def static Z postProcess(X reduction, Nd4jLong n, Z *extraParams) {
            return static_cast<Z>(reduction);
        }
    };

	template <typename X>
	class Prod {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(1);
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return opOutput * old;
		}

		op_def static X update(X old, X opOutput, X *extraParams) {
			return opOutput * old;
		}

		op_def static X op(X d1, X *extraParams) {
			return d1;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction;
		}
	};


	template <typename X, typename Z>
	class Any {
	public:
		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static X update(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, X *extraParams) {
			return d1;
		}

		op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction > static_cast<X>(0) ? static_cast<Z>(1) : static_cast<Z>(0) ;
		}
	};


    template <typename X, typename Z>
    class All {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(1);
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return opOutput * old;
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return opOutput * old;
        }

        op_def static X op(X d1, X *extraParams) {
            return d1;
        }

        op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction > static_cast<X>(0) ? static_cast<Z>(1) : static_cast<Z>(0);
        }
    };

	template <typename X, typename Z>
	class Mean {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, Z *extraParams) {
			return d1;
		}

		op_def static Z postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return (Z)( reduction / (Z) n);
		}
	};


    template <typename X, typename Z>
    class AMean {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static X merge(X old, X opOutput, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(opOutput) + nd4j::math::nd4j_abs<X>(old);
        }

        op_def static X update(X old, X opOutput, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(opOutput) + nd4j::math::nd4j_abs<X>(old);
        }

        op_def static X op(X d1, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(d1);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(reduction) / static_cast<X>(n);
        }
    };

	template <typename X>
	class Max {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return input[0];
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return nd4j::math::nd4j_max<X>(old, opOutput);
		}

		op_def static X update(X old, X opOutput, X *extraParams) {
			return nd4j::math::nd4j_max<X>(opOutput, old);
		}

		op_def static X op(X d1, X d2, X *params) {
			return nd4j::math::nd4j_max<X>(d1, d2);
		}

        op_def static X op(X d1, X d2) {
            return nd4j::math::nd4j_max<X>(d1, d2);
        }

		// FIXME: this signature overlaps with MetaOp
		op_def static X op(X d1, X *extraParams) {
			return d1;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction;
		}
	};


	template <typename X, typename Y, typename Z>
	class AMaxPairwise {
	public:
		op_def static X op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<Y>(d2));
		}

		op_def static X op(X d1, Y d2) {
			return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<Y>(d2));
		}
	};


	template <typename X, typename Y, typename Z>
	class AMinPairwise {
	public:
		op_def static X op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<Y>(d2));
		}

		op_def static X op(X d1, Y d2) {
			return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<Y>(d2));
		}
	};

	template <typename X, typename Y, typename Z>
	class MaxPairwise {
	public:
		op_def static X op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_max<X>(d1, static_cast<X>(d2));
		}

		op_def static X op(X d1, Y d2) {
			return nd4j::math::nd4j_max<X>(d1, static_cast<X>(d2));
		}
	};


	template <typename X, typename Y, typename Z>
	class MinPairwise {
	public:
		op_def static X op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_min<X>(d1, static_cast<X>(d2));
		}

		op_def static X op(X d1, Y d2) {
			return nd4j::math::nd4j_min<X>(d1, static_cast<X>(d2));
		}
	};

    template <typename X>
    class AMax {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return input[0];
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(old), nd4j::math::nd4j_abs<X>(opOutput));
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(opOutput), nd4j::math::nd4j_abs<X>(old));
        }

        op_def static X op(X d1, X d2, X *params) {
            return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<X>(d2));
        }

        op_def static X op(X d1, X d2) {
            return nd4j::math::nd4j_abs<X>(d1) > nd4j::math::nd4j_abs<X>(d2) ? d1 : d2;
        }

        // FIXME: this signature overlaps with MetaOp
        op_def static X op(X d1, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(d1);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return nd4j::math::nd4j_abs<X>(reduction);
        }
    };


	template <typename X>
	class AMin {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return input[0];
		}

		op_def static X merge(X old, X opOutput, X *extraParams) {
			return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(old), nd4j::math::nd4j_abs<X>(opOutput));
		}

		op_def static X update(X old, X opOutput, X *extraParams) {
			return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(opOutput), nd4j::math::nd4j_abs<X>(old));
		}

		op_def static X op(X d1, X d2, X *params) {
			return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<X>(d2));
		}

        op_def static X op(X d1, X d2) {
            return nd4j::math::nd4j_min<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<X>(d2));
        }

		// FIXME: this signature overlaps with MetaOp
		op_def static X op(X d1, X *extraParams) {
			return nd4j::math::nd4j_abs<X>(d1);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return nd4j::math::nd4j_abs<X>(reduction);
		}
	};

    template <typename X>
    class Min {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_cuda

        op_def static X startingValue(const X *input) {
            return input[0];
        }

        op_def static X merge(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_min<X>(old, opOutput);
        }

        op_def static X update(X old, X opOutput, X *extraParams) {
            return nd4j::math::nd4j_min<X>(opOutput, old);
        }

        op_def static X op(X d1, X d2, X *params) {
            return nd4j::math::nd4j_min<X>(d1, d2);
        }

        op_def static X op(X d1, X d2) {
            return nd4j::math::nd4j_min<X>(d1, d2);
        }

        // FIXME: this signature overlaps with MetaOp
        op_def static X op(X d1, X *extraParams) {
            return d1;
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
    };


    template <typename X, typename Z>
	class Norm1 {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static X op(X d1, Z *extraParams) {
			return nd4j::math::nd4j_abs<X>(d1);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return reduction;
		}
	};


	template <typename X, typename Z>
	class Norm2 {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_sqrt<X, X>(reduction);
		}

        op_def static X op(X d1, Z *extraParams) {
            return d1 * d1;
        }
    };

	template <typename X, typename Z>
	class SquaredNorm {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, Z *extraParams) {
			return d1 * d1;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return reduction;
		}
	};

	template <typename X, typename Z>
	class NormFrobenius {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, Z *extraParams) {
			X v = nd4j::math::nd4j_abs<X>(d1);
			return v * v;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_sqrt<X, X>(reduction);
		}
	};

	template <typename X, typename Z>
	class NormP {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static X op(X d1, Z *extraParams) {
			return nd4j::math::nd4j_pow<X, X, X>(nd4j::math::nd4j_abs(d1), extraParams[0]);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_pow<X, X, X>(reduction, static_cast<X>(1.0f) / extraParams[0]);
		}
	};

	template <typename X, typename Z>
	class NormMax {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(old),
				nd4j::math::nd4j_abs<X>(opOutput));
		}

		op_def static X op(X d1, Z *extraParams) {
			return d1;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(reduction),
				nd4j::math::nd4j_abs<X>(reduction));
		}
	};

	template <typename X, typename Z>
	class Variance {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return old + opOutput;
		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return old + opOutput;

		}

		op_def static X op(X d1, Z *extraParams) {
			X mean = extraParams[0];
			X ret = d1 - mean;
			return ret * ret;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			// T bias = extraParams[1];
			// return (reduction - (nd4j::math::nd4j_pow<T>(bias, static_cast<T>(2.0f)) / static_cast<T>(n))) / (n - 1)
			return reduction / static_cast<X>(n - 1);
		}
	};

	/**
	* Standard deviation of a buffer
	*/
	template <typename X, typename Z>
	class StandardDeviation {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X merge(X old, X opOutput, Z *extraParams) {
			return old + opOutput;
		}

		op_def static X update(X old, X opOutput, Z *extraParams) {
			return old + opOutput;

		}

		op_def static X op(X d1, Z *extraParams) {
			X mean = extraParams[0];
			X ret = d1 - mean;
			return ret * ret;
		}

		op_def static X postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			Z ret = Variance<X,Z>::postProcess(reduction, n, extraParams);
			Z sqrtRet = nd4j::math::nd4j_sqrt<X, Z>(ret);
			return sqrtRet;
		}
	};

	template <typename X, typename Y>
	class CosineSimilarity {
	public:
		static const int extraParamsLen = 2;

		op_def static X *generateExtraParams() {
			//T *extraParams = new T[2];
			return nullptr;
		}

		op_def static void finalizeExtraParams(X *extraParams) {
			//delete[] extraParams;
		}

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
			return reduction / (nd4j::math::nd4j_sqrt<X, X>(extraParams[0]) * nd4j::math::nd4j_sqrt<X, X>(extraParams[1]));
		}

		op_def static X op(X d1, Y d2, X *extraParams) {
			extraParams[0] += d1 * d1;
			extraParams[1] += d2 * d2;
			return (d1 * d2);
		}

		op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {
			extraParamsTotal[0] += extraParamsLocal[0];
			extraParamsTotal[1] += extraParamsLocal[1];
		}

#ifdef __CUDACC__
		static _CUDA_D inline X opAtomic(X d1, Y d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],static_cast<X>(d1 * d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],static_cast<X>(d2 * d2));

			return (d1 * d2);
		}
#endif

		op_def static X update(X old, X opOutput, X *extraParams) {
			return old + opOutput;
		}


		op_def static X merge(X old, X opOutput, X *extraParams) {
			return update(old, opOutput, extraParams);
		}
	};


    template <typename X, typename Y>
    class JaccardDistance {
    public:
        static const int extraParamsLen = 2;

        op_def static X *generateExtraParams() {
            //T *extraParams = new T[2];
            return nullptr;
        }

        op_def static void finalizeExtraParams(X *extraParams) {
            //delete[] extraParams;
        }

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            // num / denom
            return (static_cast<X>(1.0f)) - (extraParams[0] / extraParams[1]);
        }

        op_def static X num(X d1, Y d2) {
            return nd4j::math::nd4j_min<X>(d1, d2);
        }

        op_def static X denom(X d1, Y d2) {
            return nd4j::math::nd4j_max<X>(d1, d2);
        }

        op_def static X op(X d1, Y d2, X *extraParams) {
            extraParams[0] += num(d1, d2);
            extraParams[1] += denom(d1, d2);
            return static_cast<X>(0.0f);
        }

        op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
        __device__
		static inline X opAtomic(X d1, Y d2, X *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],num(d1, d2));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], denom(d1, d2));

			return static_cast<X>(0.0f);
		}
#endif

        op_def static  X update(X old, X opOutput, X *extraParams) {
            return old + opOutput;
        }


        op_def static X merge(X old, X opOutput, X *extraParams) {
            return update(old, opOutput, extraParams);
        }
    };


    template <typename X, typename Y>
    class SimpleHammingDistance {
    public:
        static const int extraParamsLen = 0;

        op_def static X *generateExtraParams() {
            //T *extraParams = new T[2];
            return nullptr;
        }

        op_def static void finalizeExtraParams(X *extraParams) {
            //delete[] extraParams;
        }

        op_def static X startingValue(X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return static_cast<X>(reduction / n);
        }

        op_def static X op(X d1, Y d2, X *extraParams) {
            return (d1 == static_cast<X>(d2)) ? static_cast<X>(0.0f) :  static_cast<X>(1.0f);
        }

        op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {

        }

#ifdef __CUDACC__
        __device__
		static inline X opAtomic(X d1, Y d2, X *extraParams) {
			return op(d1, d2, extraParams);
		}
#endif

        op_def static X update(X old, X opOutput, X *extraParams) {
            return old + opOutput;
        }


        op_def static X merge(X old, X opOutput, X *extraParams) {
            return update(old, opOutput, extraParams);
        }
    };

    template <typename X, typename Y>
    class CosineDistance {
    public:
        static const int extraParamsLen = 2;

        op_def static X *generateExtraParams() {
            //T *extraParams = new T[2];
            return nullptr;
        }

        op_def static void finalizeExtraParams(X *extraParams) {
            //delete[] extraParams;
        }

        op_def static X startingValue(X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return (static_cast<X>(1.0f)) - (reduction / (nd4j::math::nd4j_sqrt<X, X>(extraParams[0]) * nd4j::math::nd4j_sqrt<X, X>(extraParams[1])));
        }

        op_def static X op(X d1, Y d2, X *extraParams) {
            extraParams[0] += nd4j::math::nd4j_abs<X>(d1) * nd4j::math::nd4j_abs<X>(d1);
            extraParams[1] += nd4j::math::nd4j_abs<X>(d2) * nd4j::math::nd4j_abs<X>(d2);
            return (d1 * d2);
        }

        op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
	static _CUDA_D inline X opAtomic(X d1, Y d2, T *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0], nd4j::math::nd4j_abs<X>(d1) * nd4j::math::nd4j_abs<X>(d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], nd4j::math::nd4j_abs<Y>(d2) * nd4j::math::nd4j_abs<Y>(d2));

			return (d1 * d2);
		}
#endif

        op_def static X update(X old, X opOutput, X *extraParams) {
            return old + opOutput;
        }


        op_def static X merge(X old, X opOutput, X *extraParams) {
            return update(old, opOutput, extraParams);
        }
    };


	/**
	* Dot product between 2 arrays
	*/
	template <typename X, typename Y>
	class Dot {
	public:
		static const int extraParamsLen = 0;

		op_def static X * generateExtraParams() {
			return nullptr;
		}

		op_def static void finalizeExtraParams(X *extraParamsRef) {
			//no-op
			//delete[] * extraParamsRef;
		}

		op_def static X startingValue(X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParamsRef) {
			return reduction;
		}

		op_def static X op(X d1, Y d2, X *extraParamsRef) {
			return d1 * d2;
		}


#ifdef __CUDACC__
		__device__
		static inline X opAtomic(X d1, Y d2, X *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static X update(X old, X opOutput, X *extraParamsRef) {
			return opOutput + old;
		}

		op_def static X merge(X old, X opOutput, X *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}

		op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {}
	};


    /**
	* Op to check equality within arrays
	*/
    template <typename X, typename Y>
    class EqualsWithEps {
    public:
        static const int extraParamsLen = 0;

        op_def static X * generateExtraParams() {
            return nullptr;
        }

        op_def static void finalizeExtraParams(X *extraParamsRef) {
            //no-op
        }

        op_def static X startingValue(X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static X postProcess(X reduction, Nd4jLong n, X *extraParamsRef) {
            return reduction;
        }

        op_def static X op(X d1, Y d2, X *extraParamsRef) {

        	X eps = extraParamsRef[2];
    	    X diff = nd4j::math::nd4j_abs<X>(d1 - d2);

    		// works well except in the range of very large numbers
    		if (diff <= eps)
    	    	return static_cast<X>(0.f);

    	    // Knuth approach
    	    // works well except in the range of very small numbers
		    if (diff <= nd4j::math::nd4j_max<X>(nd4j::math::nd4j_abs<X>(d1), nd4j::math::nd4j_abs<Y>(d2)) * eps)
		    	return static_cast<X>(0.f);

        	return static_cast<X>(1.f);
        }


#ifdef __CUDACC__
        __device__
		static inline X opAtomic(X d1, Y d2, X *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

        op_def static X update(X old, X opOutput, X *extraParamsRef) {
            return opOutput + old;
        }

        op_def static X merge(X old, X opOutput, X *extraParamsRef) {
            return update(old, opOutput, extraParamsRef);
        }

        op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {}
    };



	template <typename X, typename Y>
	class EuclideanDistance {
	public:
		static const int extraParamsLen = 0;

		op_def static X * generateExtraParams() {
			return nullptr;
		}

		op_def static void finalizeExtraParams(X *extraParamsRef) {
			//no-op
		}

		op_def static X startingValue(X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParamsRef) {
			return nd4j::math::nd4j_sqrt<X, X>(reduction);
		}

		op_def static X op(X d1, Y d2, X *extraParamsRef) {
			X ret = d1 - d2;
			return ret * ret;
		}


#ifdef __CUDACC__
			__device__
			static  inline X opAtomic(X d1, Y d2, X *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static X update(X old, X opOutput, X *extraParamsRef) {
			return opOutput + old;
		}

		op_def static X merge(X old, X opOutput, X *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
		op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {}

	};


	template <typename X, typename Y>
	class ManhattanDistance  {
	public:
		static const int extraParamsLen = 0;

		op_def static X * generateExtraParams() {
			return nullptr;
	}

		op_def static void finalizeExtraParams(X *extraParamsRef) {
			//no-op
		}

		op_def static X startingValue(X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static X postProcess(X reduction, Nd4jLong n, X *extraParamsRef) {
			return reduction;
		}

		op_def static X op(X d1, Y d2, X *extraParamsRef) {
			return nd4j::math::nd4j_abs<X>(d1 - d2);
		}

		op_def static X update(X old, X opOutput, X *extraParamsRef) {
			return old + opOutput;
		}

		op_def static void aggregateExtraParams(X *extraParamsTotal, X *extraParamsLocal) {

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
		op_def static X merge(X old, X opOutput, X *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
	};


	template <typename X>
	class IndexAbsoluteMax  {
	public:
#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return nd4j::math::nd4j_abs<X>(val);
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> update(
				functions::indexreduce::IndexValue<X> old,
		functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {
			opOutput.value = nd4j::math::nd4j_abs<X>(opOutput.value);
			old.value = nd4j::math::nd4j_abs<X>(old.value);
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
		static inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
		functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (nd4j::math::nd4j_abs<X>(f1.value) > nd4j::math::nd4j_abs<X>(f2.value))
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline X startingValue(X *input) {
			return -nd4j::DataTypeUtils::max<X>();
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
		functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};

    template <typename X>
    class FirstIndex {
    public:
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
            return val;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<X> update(
                functions::indexreduce::IndexValue<X> old,
                functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {

#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            auto res = simdOps::MatchCondition<X>::op(opOutput.value, extraParams);

			//printf("res: %f; oldIdx: %i; newIdx: %i\n", res, old.index, opOutput.index);

            if (res == static_cast<X>(0))
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
        static inline X startingValue(X *input) {
            return -nd4j::DataTypeUtils::max<X>();
        }


#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                               functions::indexreduce::IndexValue<X> d2, X *extraParams) {
            return d1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> merge(
                functions::indexreduce::IndexValue<X> f1,
                functions::indexreduce::IndexValue<X> f2, X *extraParams) {
            if (f1.index > f2.index)
                return f2;
            return f1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> postProcess(
                functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
                X *dx, int incx, X *extraParams, X *result) {
            return reduction;
        }
    };


    template <typename X>
    class LastIndex {
    public:
#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
            return val;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<X> update(
                functions::indexreduce::IndexValue<X> old,
                functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {
#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            auto res = simdOps::MatchCondition<X>::op(opOutput.value, extraParams);

            if (res == static_cast<X>(0))
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
        static inline X startingValue(X *input) {
            return -nd4j::DataTypeUtils::max<X>();
        }


#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                               functions::indexreduce::IndexValue<X> d2, X *extraParams) {
            return d1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> merge(
                functions::indexreduce::IndexValue<X> f1,
                functions::indexreduce::IndexValue<X> f2, X *extraParams) {
            if (f1.index < f2.index)
                return f2;
            return f1;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> postProcess(
                functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
                X *dx, int incx, X *extraParams, X *result) {
            return reduction;
        }
    };


	template <typename X>
	class IndexMax  {
	public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static functions::indexreduce::IndexValue<X> update(
				functions::indexreduce::IndexValue<X> old,
				functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {
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
        static inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
				functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (f1.value > f2.value)
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline X startingValue(X *input) {
			return -nd4j::DataTypeUtils::max<X>();
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
				functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};


	template <typename X>
	class IndexAbsoluteMin {
	public:
#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> op(
				functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline X startingValue(X *input) {
			return nd4j::DataTypeUtils::max<X>();
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> update(
				functions::indexreduce::IndexValue<X> old,
		functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {
			opOutput.value = nd4j::math::nd4j_abs<X>(opOutput.value);
			old.value = nd4j::math::nd4j_abs<X>(old.value);
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
		static inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
		functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (nd4j::math::nd4j_abs<X>(f1.value) < nd4j::math::nd4j_abs<X>(f2.value))
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
		static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
		functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};


	template <typename X>
	class IndexMin {
	public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(
				functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline X startingValue(X *input) {
			return nd4j::DataTypeUtils::max<X>();
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> update(
				functions::indexreduce::IndexValue<X> old,
				functions::indexreduce::IndexValue<X> opOutput, X *extraParams) {
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
        static inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
				functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (f1.value < f2.value)
				return f2;
			return f1;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

#ifdef __CUDACC__
        __host__ __device__
#endif
        static inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
				functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};

	template <typename X>
	class SummaryStatsVariance {
	public:

        static _CUDA_HD inline X getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
			if (biasCorrected) {
				X ret = val.varianceBiasCorrected();
				if (ret < static_cast<X>(0.0f))
					return val.variance();
				return ret;
			}
			return val.variance();
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<X> op(functions::summarystats::SummaryStatsData<X> d1, X *extraParams) {
			return d1;
		}
	};

	template <typename X>
	class SummaryStatsStandardDeviation {
	public:

        static _CUDA_HD inline X getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
			if (biasCorrected) {
				X ret = val.varianceBiasCorrected();
				if (ret < static_cast<X>(0.0f))
					return nd4j::math::nd4j_sqrt<double, X>(val.variance());
				else
					return nd4j::math::nd4j_sqrt<X, X>(ret);
			}
			return  nd4j::math::nd4j_sqrt<double, X>(val.variance());
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<X> op(functions::summarystats::SummaryStatsData<X> d1, X *extraParams) {
			return d1;
		}
	};

    template <typename X>
	class DropOut {
	public:
		no_op_exec_special_same
		no_op_exec_special_cuda

		inline _CUDA_D static X op(X d1, X *params) {
			X prob = params[0];

#ifdef __CUDACC__
			X length = params[1];
            X tid = gridDim.x * blockDim.x + threadIdx.x;
            X rnd = nd4j::math::nd4j_abs<X>(nd4j::math::nd4j_cos<X>(static_cast<X>(clock64()) * static_cast<X>(tid) + static_cast<X>(length) * static_cast<X>(tid)));
#else
			X rnd = static_cast<X>(rand() / RAND_MAX);
#endif
			return rnd >= prob ? static_cast<X>(0.0f) : d1;
		}
	};

    template <typename X, typename Y, typename Z>
	class DropOutInverted {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

#ifdef __CUDACC__
    __device__
#endif
        inline static Z op(X d1, Y d2, Z *params) {
			Y prob = d2;
#ifdef __CUDACC__
			X length = params[1];
			X tid = gridDim.x * blockDim.x + threadIdx.x;
            X rnd = nd4j::math::nd4j_abs<X>(nd4j::math::nd4j_cos<X>(static_cast<X>(clock64()) * static_cast<X>(tid) + static_cast<X>(length) * static_cast<X>(tid)));
#else
			X rnd = static_cast<X>(rand() / RAND_MAX);
#endif
			return rnd >= static_cast<X>(prob) ? static_cast<Z>(0.0f) : reinterpret_cast<Z>(d1 / static_cast<X>(prob));
		}
	};


	template <typename X, typename Y, typename Z>
	class ReplaceNans {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_isnan(d1) ? static_cast<Z>(d2) : static_cast<Z>(d1) ;
		}
	};

    // this op is used for conditional pairwise transforms only
    template <typename X, typename Y, typename Z>
    class CompareAndReplace{
    public:
        // op definition for PairWise Transform
        op_def static Z op(X d1, Y d2, Z *params) {
			auto compare = params[0];
            auto eps = params[2];
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<X>(d1 - compare) <= eps)
                    return d2;
                else
                    return d1;
            else if (mode == 1) // not equals eps
                if (nd4j::math::nd4j_abs<X>(d1 - compare) > eps)
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
                if (nd4j::math::nd4j_abs<X>(d1) < compare)
                    return d2;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<X>(d1) > compare)
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
                if (nd4j::math::nd4j_abs<X>(d1) >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<X>(d1) <= compare)
                    return d2;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
        }
    };

	template <typename X, typename Y, typename Z>
	class CompareAndSet {
	public:
        // op definition for Transform
		op_def static X op(X d1, X *params) {
			auto compare = params[0];
			auto set = params[1];
			auto eps = params[2];

            // with mode == 0 we do set if d1 equals to compare, and with mode == 1 - we go otherwise
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<X>(d1 - compare) <= eps)
                    return set;
				else
                    return d1;
			    //return nd4j::math::nd4j_abs<T>(d1 - compare) <= eps ? set : d1;
            else if (mode == 1) // not equals
                if (nd4j::math::nd4j_abs<X>(d1 - compare) > eps)
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
                if (nd4j::math::nd4j_abs<X>(d1) < compare)
                    return set;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<X>(d1) > compare)
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
                if (nd4j::math::nd4j_abs<X>(d1) >= compare)
                    return set;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<X>(d1) <= compare)
                    return set;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
		}

        // op definition for PairWise Transform
        op_def static Z op(X d1, Y dY, Z *params) {
		    X d2 = static_cast<X>(dY);
            auto compare = params[0];
			auto eps = params[2];
			auto mode = static_cast<int>(params[3]);
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<X>(d2 - compare) <= eps)
                    return d2;
                else
                    return d1;
            else if (mode == 1) // not equals
                if (nd4j::math::nd4j_abs<X>(d2 - compare) > eps)
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
                if (nd4j::math::nd4j_abs<X>(d2) < compare)
                    return d2;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<X>(d2) > compare)
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
                if (nd4j::math::nd4j_abs<X>(d1) >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<X>(d1) <= compare)
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
	
