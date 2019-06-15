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
#include <loops/ReduceType.h>

#define MIN_V 1e-12
#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#define MAX_INT 2147483647
#define MIN_CUTFOFF -3.79297773665f
#define FLOAT_MIN_NORMAL 1.17549435e-38
#define EPS 1e-5
#define AFFINITY close
#define DOUBLE_PI_T T(2.0 * 3.14159265358979323846)
#define DOUBLE_PI_X X(2.0 * 3.14159265358979323846)

#define no_op_exec_special_any 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_bool 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_same 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, X *result, Nd4jLong *resultShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special 	static const bool requiresSpecial = false; static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, Z *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation 	static const bool requiresSpecialAccumulation = false; static void execSpecial(X *x, Nd4jLong *xShapeInfo, Z *extraParams, Z *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
#define no_op_exec_special_accumulation_long 	static const bool requiresSpecialAccumulation = false; static void execSpecial(X *x, Nd4jLong *xShapeInfo, X *extraParams, Z *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
#define no_op_exec_special_accumulation_same 	static const bool requiresSpecialAccumulation = false; static void execSpecial(X *x, Nd4jLong *xShapeInfo, X *extraParams, X *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset){}
#ifdef __CUDACC__
#define no_op_exec_special_any_cuda static __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, X *extraParams, int *allocationPointer, Z *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_bool_cuda static __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeBuffer, Z *result, Nd4jLong *resultShapeBuffer, X *extraParams, int *allocationPointer, Z *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_same_cuda static __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeBuffer, X *result, Nd4jLong *resultShapeBuffer, X *extraParams, int *allocationPointer, X *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_cuda static __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeBuffer,Z *result, Nd4jLong *resultShapeBuffer,Z *extraParams, int *allocationPointer, Z *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation_same_cuda static inline __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeInfo, X *extraParams, X *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, X *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation_long_cuda static inline __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeInfo, X *extraParams, Z *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Z *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {}
#define no_op_exec_special_accumulation_cuda static inline __device__ void execSpecialCuda(X *dx, Nd4jLong *xShapeInfo, Z *extraParams, Z *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Z *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {}

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
#define no_op_exec_special_accumulation_same_cuda
#define no_op_exec_special_accumulation_long_cuda
#define no_op_exec_special_any_cuda
#define no_op_exec_special_bool_cuda
#define no_op_exec_special_same_cuda
#define no_op_exec_special_accumulation_same_cuda
#endif


#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_LAMBDA 1.0507009873554804934193349852946

#ifdef _OPENMP
#pragma omp declare reduction(maxTF : float,double,float16,bfloat16 :              \
                omp_out = nd4j::math::nd4j_max(omp_in, omp_out) )\
                initializer (omp_priv=-MAX_FLOAT)

#pragma omp declare reduction(minTF : float,double,float16,bfloat16 :              \
                omp_out = nd4j::math::nd4j_min(omp_in, omp_out) )\
                initializer (omp_priv=MAX_FLOAT)

#pragma omp declare reduction(maxT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = nd4j::math::nd4j_max(omp_in, omp_out) )\
                initializer (omp_priv=0)

#pragma omp declare reduction(minT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = nd4j::math::nd4j_min(omp_in, omp_out) )\
                initializer (omp_priv=0)

#pragma omp declare reduction(amaxT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = nd4j::math::nd4j_max(nd4j::math::nd4j_abs(omp_in), nd4j::math::nd4j_abs(omp_out)) )

#pragma omp declare reduction(aminT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = nd4j::math::nd4j_min(nd4j::math::nd4j_abs(omp_in), nd4j::math::nd4j_abs(omp_out)) )

#pragma omp declare reduction(asumT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = nd4j::math::nd4j_abs(omp_in) + nd4j::math::nd4j_abs(omp_out))\
                initializer (omp_priv=0)

#pragma omp declare reduction(sumT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = omp_in + omp_out)\
                initializer (omp_priv=0)

#pragma omp declare reduction(prodT : float,double,float16,bfloat16,int,Nd4jLong,Nd4jULong,int8_t,uint8_t,bool,int16_t,uint16_t,uint32_t :              \
                omp_out = omp_in * omp_out)\
                initializer (omp_priv=1)
#endif


namespace functions {
	namespace indexreduce {
		template <typename T>
		struct IndexValue {
			T value;
            Nd4jLong index;
            _CUDA_HD IndexValue() = default;
			_CUDA_HD IndexValue(const T val, const Nd4jLong ind): index(ind), value(val) {}
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
			return static_cast<Z>(d1 + d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d1 + d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1 + params[0]);
		}

		op_def static X startingValue() {
			return static_cast<X>(0.f);
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
			return static_cast<Z>(d1 - d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d1 - d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1 - params[0]);
		}

	};

    template <typename X, typename Y, typename Z>
    class SquaredSubtract {
    public:
        op_def static Z op(X d1, Y d2) {
            auto d = static_cast<Z>(d1 - d2);
            return d * d;
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto d = static_cast<Z>(d1 - d2);
            return d * d;
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            auto d = static_cast<Z>(d1 - params[0]);
            return d * d;
        }
    };

	template <typename X, typename Y, typename Z>
	class SquaredReverseSubtract {
	public:
        op_def static Z op(X d1, Y d2) {
            auto d = static_cast<Z>(d2 - d1);
            return d * d;
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto d = static_cast<Z>(d2 - d1);
            return d * d;
        }

        op_def static Z op(X d1) {
            return d1;
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            auto d = static_cast<Z>(params[0] - d1);
            return d * d;
        }
	};

	template <typename X, typename Y, typename Z>
	class ReverseSubtract {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d2 - d1);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d2 - d1);
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(params[0] - d1);
		}
	};


	template <typename X, typename Y, typename Z>
	class LogPoissonLossFull {

	public:
		op_def static Z op(X z, Y c) {
			auto zz = static_cast<Z>(z);
			auto zc = static_cast<Z>(c);
			return (nd4j::math::nd4j_exp<Y, Z>(c) - zz * zc  + (zz * nd4j::math::nd4j_log<X, Z>(z) - zz + static_cast<Z>(0.5f) * nd4j::math::nd4j_log<Z, Z>(static_cast<Z>(DOUBLE_PI_X) * zz)));
		}

		op_def static Z op(X z, Y c, Z *params) {
			auto zz = static_cast<Z>(z);
			auto zc = static_cast<Z>(c);
			return (nd4j::math::nd4j_exp<Y, Z>(c) - zz * zc  + (zz * nd4j::math::nd4j_log<X, Z>(z) - zz + static_cast<Z>(0.5f) * nd4j::math::nd4j_log<Z, Z>(static_cast<Z>(DOUBLE_PI_X) * zz)));
		}

		op_def static Z op(X z) {
			auto zz = static_cast<Z>(z);
			return (zz * nd4j::math::nd4j_log<Y, Z>(z) - zz + static_cast<Z>(0.5f) * nd4j::math::nd4j_log<Z, Z>(static_cast<Z>(DOUBLE_PI_X) * zz));
		}

		// op for MetaOps
		op_def static X op(X z, Y *params) {
			return (nd4j::math::nd4j_exp<X, X>(params[0]) - z * params[0]  + (z * nd4j::math::nd4j_log<X, Z>(z) - z + static_cast<X>(0.5f) * nd4j::math::nd4j_log<X, Z>(DOUBLE_PI_X * z)));
		}
	};

	template <typename X, typename Y, typename Z>
	class LogPoissonLoss {

	public:
		op_def static Z op(X z, Y c) {
			auto zz = static_cast<Z>(z);
			auto zc = static_cast<Z>(c);
			return (nd4j::math::nd4j_exp<Y, Z>(c) - zz * zc);
		}

		op_def static Z op(X z, Y c, Z *params) {
			auto zz = static_cast<Z>(z);
			auto zc = static_cast<Z>(c);
			return (nd4j::math::nd4j_exp<Y, Z>(c) - zz * zc);
		}

		op_def static Z op(X z) {
			return static_cast<Z>(z);
		}

		// op for MetaOps
		op_def static Z op(X z, Y *params) {
			return (nd4j::math::nd4j_exp<Y, Z>(params[0]) - static_cast<Z>(z) * static_cast<Z>(params[0]));
		}
	};

	template <typename X, typename Y, typename Z>
	class Multiply {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d1 * d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d1 * d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1 * params[0]);
		}

		op_def static X startingValue() {
			return static_cast<X>(1.f);
		}
	};

	template <typename X, typename Y, typename Z>
	class Divide {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d1 / d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d1 / d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1 / params[0]);
		}

		op_def static X startingValue() {
			return static_cast<X>(1);
		}
	};

	template <typename X, typename Y, typename Z>
	class SafeDivide {
	public:
		op_def static Z op(X d1, Y d2) {
			if(d2 == static_cast<Y>(0))
				return static_cast<Z>(0);
			return static_cast<Z>(d1 / d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			if(d2 == static_cast<Y>(0))
				return static_cast<Z>(0);
			return static_cast<Z>(d1 / d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			if(params[0] == static_cast<Y>(0))
				return static_cast<Z>(0);
			return static_cast<Z>(d1 / params[0]);
		}
	};

    template <typename X, typename Y, typename Z>
    class FloorDiv {
    public:
        op_def static Z op(X d1, Y d2) {
            return nd4j::math::nd4j_floor<Z,Z>(static_cast<Z>(d1 / d2));
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            return nd4j::math::nd4j_floor<Z,Z>(static_cast<Z>(d1 / d2));
        }

        op_def static Z op(X d1) {
            return nd4j::math::nd4j_floor<Z,Z>(static_cast<Z>(d1));
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            return nd4j::math::nd4j_floor<Z,Z>(static_cast<Z>(d1 / params[0]));
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
    class TruncateMod {
    public:
        op_def static Z op(X d1, Y d2) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<Z>(i1 % i2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(d2);
            return static_cast<Z>(i1 % i2);
        }

        op_def static Z op(X d1) {
            return static_cast<Z>(d1);
        }

        // op for MetaOps
        op_def static Z op(X d1, Y *params) {
            auto i1 = static_cast<int>(d1);
            auto i2 = static_cast<int>(params[0]);
            return static_cast<Z>(i1 % i2);
        }
    };

    template<typename X, typename Y, typename Z>
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
			auto m = nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);
            return (d1 < static_cast<X>(0)) == (d2 < static_cast<Y>(0)) ? m : nd4j::math::nd4j_fmod<Z, Y, Z>(m + static_cast<Z>(d2), d2);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            auto m = nd4j::math::nd4j_fmod<X, Y, Z>(d1, d2);
			return (d1 < static_cast<X>(0.0f)) == (d2 < static_cast<Y>(0)) ? m : nd4j::math::nd4j_fmod<Z, Y, Z>(m + static_cast<Z>(d2), d2);
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
			return static_cast<Z>(d2 / d1);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d2 / d1);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		// op for MetaOps
		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(params[0] / d1);
		}
	};

	template <typename X, typename Y, typename Z>
	class CopyPws {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1);
		}
	};

	template <typename X>
	class Copy {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X, typename Y, typename Z>
	class Copy2 {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d2);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}

		op_def static Z op(X d1, Y *params) {
			return static_cast<Z>(d1);
		}
	};

	template <typename X, typename Y, typename Z>
	class Axpy {
	public:
		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d2 + d1);
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			auto alpha = params[0];
			return alpha * static_cast<Z>(d1) + static_cast<Z>(d2);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(d1);
		}
	};

	template <typename X, typename Z>
	class Assign {
	public:
		no_op_exec_special_any
		no_op_exec_special_any_cuda

		op_def static Z op(X d1, X *params) {
			return static_cast<Z>(d1);
		}
	};

	template <typename X, typename Z>
	class And {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		op_def static Z op(X d1, X d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, X d2, X *params) {
		    if (params != nullptr) {
                auto comp = params[0];
                return d1 != comp && d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
            } else {
                auto b1 = static_cast<bool>(d1);
                auto b2 = static_cast<bool>(d2);

                return (b1 && b2) ? static_cast<Z>(1) : static_cast<Z>(0);
		    }
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, X *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Z>
	class Or {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		op_def static Z op(X d1, X d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, X d2, X *params) {
		    if (params != nullptr) {
                auto comp = params[0];

                return d1 != comp || d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
            } else {
                auto b1 = static_cast<bool>(d1);
                auto b2 = static_cast<bool>(d2);

                return b1 || b2 ? static_cast<Z>(1) : static_cast<Z>(0);
		    }
		}

		op_def static Z op(X d1) {
			return d1;
		}

		// op for MetaOps
		op_def static Z op(X d1, X *params) {
			return static_cast<Z>(119);
		}
	};

	template <typename X, typename Z>
	class Xor {
	public:

		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		op_def static Z op(X d1, X d2) {
			return d2 + d1;
		}

		op_def static Z op(X d1, X d2, X *params) {
			if (params != nullptr) {
                auto comp = params[0];

                return ((d1 == comp && d2 != comp) || (d1 != comp && d2 == comp)) ? static_cast<Z>(1) : static_cast<Z>(0);
            } else {
                auto b1 = static_cast<bool>(d1);
                auto b2 = static_cast<bool>(d2);

                return (!b1 && b2 )||(b1 && !b2) ? static_cast<Z>(1) : static_cast<Z>(0);
			}
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};


	template <typename X, typename Z>
	class Not {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		op_def static Z op(X d1, X d2) {
            return static_cast<Z>(0);
		}

		op_def static Z op(X d1, X d2, X *params) {
			return d1 != d2 ? static_cast<Z>(1) : static_cast<Z>(0);
		}

		// this transform op should run only on boolean input
        op_def static Z op(X d1, X *params) {
		    auto b1 = static_cast<bool>(d1);
            return !b1;
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
	template <typename X, typename Z>
	class Epsilon {
	public:

	    op_def static Z op(X d1, X d2) {
            X diff = d1 - d2;
            X absDiff = nd4j::math::nd4j_abs<X>(diff);
            if (absDiff <= static_cast<X>(MIN_V))
                return static_cast<Z>(1);
            return static_cast<Z>(0);
	    }

		op_def static Z op(X d1, X d2, X *params) {
            return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X, typename Z>
	class EqualTo {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 == d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X, typename Z>
	class NotEqualTo {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 != d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};



	template <typename X, typename Z>
	class GreaterThanOrEqual {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 >= d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
			return op(d1, d2);
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static Z op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X, typename Z>
	class GreaterThan {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 > d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
			return op(d1, d2);
		}

		// FIXME: this signature clashes with MetaOp stuff
		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X, typename Z>
	class LessThan {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 < d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, X *params) {
			return d1;
		}

	};


	template <typename X, typename Z>
	class LessThanOrEqual {
	public:
		op_def static Z op(X d1, X d2) {
			return d1 <= d2;
		}

		op_def static Z op(X d1, X d2, X *params) {
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
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_abs<X>(d1);
		}
	};


	template <typename X>
	class Ceiling {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_ceil<X,X>(d1);
		}
	};


	template <typename X>
	class Cosine {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_cos<X,X>(d1);
		}
	};


	template <typename X>
	class Exp {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_exp<X, X>(d1);
		}
	};


	template <typename X>
	class HardTanhDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return ((d1 >= static_cast<X>(-1.f) && d1 <= static_cast<X>(1.f)) ? static_cast<X>(1.f) : static_cast<X>(0.f));
		}
	};


	template <typename X>
	class HardTanh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			if (d1 < static_cast<X>(-1))
				return static_cast<X>(-1);
			else if (d1 > static_cast<X>(1))
				return static_cast<X>(1);
			else
				return d1;

		}
	};


	template <typename X>
	class Floor {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_floor<X,X>(d1);
		}
	};


	template <typename X>
	class Log {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_log<X, X>(d1);
		}
	};

	template <typename X>
	class Log1p {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_log<X, X>(1 + d1);
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
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            if (d1 <= static_cast<X>(0))
            	return static_cast<X>(nd4j::DataTypeUtils::min<float16>());
            else return d1;
        }
    };

    template <typename X>
    class StabilizeX {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            if (d1 <= static_cast<X>(0))
            	return nd4j::DataTypeUtils::min<X>();
            else return d1;
        }
    };

	template <typename X>
	class SpecialDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return d1 * (static_cast<X>(1.f) - d1);
		}
	};


	template <typename X>
	class Neg {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return -d1;
		}
	};

	template <typename X>
	class Erf {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_erf<X,X>(d1);
		}
	};


	template <typename X>
	class Erfc {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_erfc<X,X>(d1);
		}
	};

	template <typename X>
	class Reciprocal {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda
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

		op_def static Z op(X d1) {
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
			return nd4j::math::nd4j_re<X>(d1, d2) > threshold ? static_cast<Z>(1) : static_cast<Z>(0);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(0);
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
			return nd4j::math::nd4j_re<X>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0) : static_cast<Z>(1)) : static_cast<Z>(0);
 		}

		op_def static Z op(X d1, Y d2, Z *params) {
			X thresholdRelative = params[0];
			X thresholdAbsolute = params[1];
			return nd4j::math::nd4j_re<X>(d1, d2) > thresholdRelative ? (nd4j::math::nd4j_abs<X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0) : static_cast<Z>(1)) : static_cast<Z>(0);
		}

		op_def static Z op(X d1) {
			return static_cast<Z>(0);
		}
	};

    template <typename X, typename Y, typename Z>
    class ReversePow {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        op_def static Z op(X d1, Z *params) {
            return nd4j::math::nd4j_pow<X, X, Z>(params[0], d1);
        }

        op_def static Z op(X d1, Y d2) {
            return nd4j::math::nd4j_pow<X, Y, Z>(d2, d1);
        }

        op_def static Z op(X d1, Y d2, Z *params) {
            return nd4j::math::nd4j_pow<X, Y, Z>(d2, d1);
        }

        op_def static Z op(X d1) {
            return d1;
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
			return params[0] * nd4j::math::nd4j_pow<X, Z, Z>(d1, static_cast<Z>(params[0]) - static_cast<Z>(1.f));
		}

		op_def static Z op(X d1, Y d2) {
			return static_cast<Z>(d2) * nd4j::math::nd4j_pow<X, Z, Z>(d1, static_cast<Z>(d2) - static_cast<Z>(1.f));
		}

		op_def static Z op(X d1, Y d2, Z *params) {
			return static_cast<Z>(d2) * nd4j::math::nd4j_pow<X, Z, Z>(d1, static_cast<Z>(d2) - static_cast<Z>(1.f));
		}

		op_def static Z op(X d1) {
			return d1;
		}
	};


	template <typename X>
	class Round {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_round<X,X>(d1);
		}
	};

	template <typename X, typename Z>
	class IsNan {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isnan(d1) ? static_cast<X>(1) : static_cast<X>(0);
		}

        op_def static X startingValue(const X *input) {
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


	template <typename X>
	class Expm1 {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_exp<X, X>(d1) - static_cast<X>(1);
		}
	};

	template <typename X, typename Z>
	class IsPositive {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return d1 > (X)0.f;
		}

        op_def static X startingValue(const X *input) {
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
	class IsInf {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isinf<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
		}

        op_def static X startingValue(const X *input) {
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
	class IsInfOrNan{
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isfin<X>(d1) ? static_cast<Z>(0) : static_cast<Z>(1);
		}

		op_def static X startingValue(const X *input) {
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
	class IsFinite {
	public:
		no_op_exec_special_bool
		no_op_exec_special_bool_cuda

		no_op_exec_special_accumulation
		no_op_exec_special_accumulation_cuda

		op_def static Z op(X d1, X *params) {
			return nd4j::math::nd4j_isfin<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
		}

        op_def static X startingValue(const X *input) {
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


	template <typename X>
	class ClipByValue {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			if (d1 > params[1])
				return params[1];
			if (d1 < params[0])
				return params[0];
			return d1;
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
			else if (d1 < -_v)
				return -_v;
			else return d1;
		}
	};

	template <typename X>
    class Swish {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 * nd4j::math::nd4j_sigmoid<X,X>(d1);
        }
    };

    template <typename X>
    class GELU {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 * nd4j::math::nd4j_sigmoid<X,X>(static_cast<X>(1.702f) * d1);
        }
    };

    template <typename X>
    class PreciseGELU {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            auto sp = nd4j::math::nd4j_sqrt<X, X>(static_cast<X>(2) / static_cast<X>(M_PI));
            auto xp = d1 + nd4j::math::nd4j_pow<X, X, X>(static_cast<X>(0.044715) * d1, static_cast<X>(3));
            return (d1 / static_cast<X>(2)) * (static_cast<X>(1) + nd4j::math::nd4j_tanh<X, X>(sp * xp));
        }
    };

    template <typename X>
    class GELUDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            auto x17 = static_cast<X>(1.702f) * d1;
            auto ep = nd4j::math::nd4j_pow<X,X,X>(static_cast<X>(M_E), x17);
            // (E^(1.702 x) (1. + E^(1.702 x) + 1.702 x))/(1. + E^(1.702 x))^2
            return (ep * (static_cast<X>(1.f) + ep + x17)) / nd4j::math::nd4j_pow<X, int, X>((static_cast<X>(1.f) + ep), 2);
        }
    };

    template <typename X>
    class PreciseGELUDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            auto x79 = static_cast<X>(0.797885) * d1;
            auto x03 = nd4j::math::nd4j_pow<X, int, X>(static_cast<X>(0.0356774) * d1, 3);
            auto x39 = static_cast<X>(0.398942) * d1;
            auto x05 = nd4j::math::nd4j_pow<X, int, X>(static_cast<X>(0.0535161) * d1, 3);
            auto scz = nd4j::math::nd4j_sech<X, X>(x79 + x03);
            // 0.5 + (0.398942 x + 0.0535161 x^3) Sech[0.797885 x + 0.0356774 x^3]^2 + 0.5 Tanh[0.797885 x + 0.0356774 x^3]
            return static_cast<X>(0.5) + (x39 + x05) * (scz * scz) + static_cast<X>(0.5) * nd4j::math::nd4j_tanh<X, X>(x79 + x03);
        }
    };


	template <typename X>
	class SwishDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			X ex = nd4j::math::nd4j_pow<X, X, X>(static_cast<X>(M_E), d1);
			return (ex * (d1 + ex + static_cast<X>(1.f))) / nd4j::math::nd4j_pow<X, X, X>((ex + static_cast<X>(1.f)) , static_cast<X>(2.f));
		}
	};


	template <typename X>
	class LogSigmoid {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_log<X, X>(nd4j::math::nd4j_sigmoid<X, X>(d1));
		}
	};

	template <typename X>
	class LogSigmoidDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			X ex = nd4j::math::nd4j_pow<X, X, X>(M_E, d1);
			return static_cast<X>(1.f) / (ex + static_cast<X>(1.f));
		}
	};

	template <typename X>
	class Sigmoid {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_sigmoid<X, X>(d1);
		}
	};

	template <typename X>
	class SigmoidDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_sigmoidderivative<X, X>(d1);
		}
	};


    template <typename X>
    class HardSigmoid {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return nd4j::math::nd4j_min<X>(static_cast<X>(1), nd4j::math::nd4j_max<X>(static_cast<X>(0), (static_cast<X>(0.2f)) * d1 + static_cast<X>(0.5f)));
        }
    };

    template <typename X>
    class HardSigmoidDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 < static_cast<X>(-2.5f) || d1 > static_cast<X>(2.5f) ? static_cast<X>(0.f) : static_cast<X>(0.2f);
        }
    };


	/**
	* Scale to be between a min and max
	*/
	template <typename X>
	class SetRange {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			auto min = params[0];
			auto max = params[1];
			if (static_cast<X>(d1) >= min && static_cast<X>(d1) <= max)
				return d1;
			if (min == static_cast<X>(0) && max == static_cast<X>(1)) {
				auto val = static_cast<X>(1) / (static_cast<X>(1) + nd4j::math::nd4j_exp<X, X>(-d1));
				return (nd4j::math::nd4j_floor<X,X>(val * (max - min)) + min);
			}

			return (nd4j::math::nd4j_floor<X,X>(d1 * (max - min)) + min);
		}
	};


	template <typename X>
	class Sin {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_sin<X,X>(d1);
		}
	};

	template <typename X>
	class Square {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

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

	template <typename X>
	class Rint {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_rint<X,X>(d1);
		}
	};


	template <typename X>
	class SoftPlus {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::softplus<X, X>(d1);
		}
	};


	template <typename X>
	class Sign {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return (d1 > static_cast<X>(0)) - (d1 < static_cast<X>(0));
		}
	};


	template <typename X>
	class TimesOneMinus {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return d1 * (static_cast<X>(1) - d1);
		}
	};


	template <typename X>
	class RationalTanh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			// keep 2/3 as runtime variable, to match precision
			auto dis = (static_cast<X>(2) / static_cast<X>(3)) * d1;

			auto tanh = nd4j::math::nd4j_sgn<X,X>(dis) * (static_cast<X>(1) - (static_cast<X>(1) / (static_cast<X>(1) + static_cast<X>(nd4j::math::nd4j_abs<X>(dis)) + nd4j::math::nd4j_pow<X, X, X>(dis, static_cast<X>(2)) + static_cast<X>(1.41645f) * nd4j::math::nd4j_pow<X, X, X>(dis, static_cast<X>(4)) )));
			return static_cast<X>(1.7159f) * tanh;
		}
	};

	template <typename X>
	class RationalTanhDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			auto dis = (static_cast<X>(2.f) / static_cast<X>(3.f)) * d1;

			auto a = static_cast<X>(1.f) + nd4j::math::nd4j_abs<X>(dis) + nd4j::math::nd4j_pow<X, X, X>(dis, static_cast<X>(2.f)) + static_cast<X>(1.41645f) * nd4j::math::nd4j_pow<X, X, X>(dis, static_cast<X>(4));

			auto tDeriv = (static_cast<X>(1.f) + nd4j::math::nd4j_sign<X,X>(dis) * (static_cast<X>(2.f) * dis + static_cast<X>(4.f) * static_cast<X>(1.41645f) * nd4j::math::nd4j_pow<X, X, X>(dis, static_cast<X>(3)))) / (a * a);

			return static_cast<X>(1.7159f) * (static_cast<X>(2.f) / static_cast<X>(3.f)) * tDeriv;
		}
	};

	template <typename X>
	class Tanh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_tanh<X, X>(d1);
		}
	};

    template <typename X>
    class RectifiedTanh {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return nd4j::math::nd4j_max<X>(static_cast<X>(0), nd4j::math::nd4j_tanh<X,X>(d1));
        }
    };

    template <typename X>
    class RectifiedTanhDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 > static_cast<X>(0.f) ? nd4j::math::nd4j_tanhderivative<X,X>(d1) : static_cast<X>(0.f);
        }
    };

	template <typename X>
	class ATanh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_atanh<X,X>(d1);
		}
	};

	template <typename X>
	class TanhDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_tanhderivative<X,X>(d1);
		}
	};

	template <typename X>
	class Cube {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return d1 * d1 * d1;
		}
	};


	template <typename X>
	class CubeDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

			op_def static X op(X d1, X *params) {
			return static_cast<X>(3) * d1 * d1;
		}
	};

	template <typename X>
	class ACos {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_acos<X, X>(d1);
		}
	};

	template <typename X>
	class ASinh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_asinh<X, X>(d1);
		}
	};

	template <typename X>
	class ASinhDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1.f) / (nd4j::math::nd4j_sqrt<X, X>(nd4j::math::nd4j_pow<X, X, X>(d1, static_cast<X>(2.f)) + static_cast<X>(1.f)));
		}
	};

	template <typename X>
	class ACosh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_acosh<X, X>(d1);
		}
	};


	template <typename X>
	class ACoshDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1.f) / (nd4j::math::nd4j_sqrt<X, X>(d1 - static_cast<X>(1.f)) * nd4j::math::nd4j_sqrt<X, X>(d1 + static_cast<X>(1.f)));
		}
	};



	template <typename X>
	class Ones {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1.0f);
		}
	};



	template <typename X>
	class SoftSign {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_softsign<X, X>(d1);
		}
	};


	template <typename X>
	class SoftSignDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_softsignderivative<X,X>(d1);
		}
	};

    template <typename X, typename Z>
    class MatchConditionBool {
    public:
        no_op_exec_special_bool
        no_op_exec_special_bool_cuda

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
                case 14:
                    // isFinite
                    return !(nd4j::math::nd4j_isinf(d1) || nd4j::math::nd4j_isnan(d1));
                case 15:
                    // isInfinite
                    return nd4j::math::nd4j_isinf(d1) || nd4j::math::nd4j_isnan(d1);
                default:
                    printf("Undefined match condition: [%i]\n", mode);
            }

            return d1;
        }
    };

    template <typename X, typename Z>
    class MatchCondition {
    public:
		no_op_exec_special
		no_op_exec_special_cuda

		no_op_exec_special_accumulation_long
        no_op_exec_special_accumulation_cuda

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0);
        }

        op_def static Z merge(Z old, Z opOutput, X *extraParams) {
            return old + opOutput;
        }

        op_def static Z update(Z old, Z opOutput, X *extraParams) {
            return old + opOutput;
        }

        // this op return 1.0 if condition met, 0.0 otherwise
        op_def static Z op(X d1, X *extraParams) {
            X compare = extraParams[0];
            X eps = extraParams[1];

            auto mode = static_cast<int>(extraParams[2]);
            //printf("value: %f; comp: %f; eps: %f; mode: %i;\n", (float) d1, (float) compare, (float) eps, mode);

			switch (mode) {
				case 0: // equals
					return nd4j::math::nd4j_abs<X>(d1 - compare) <= eps ? 1 : 0;
				case 1: // not equals
					return nd4j::math::nd4j_abs<X>(d1 - compare) > eps ? 1 : 0;
				case 2: // less_than
					return d1 < compare ? 1 : 0;
				case 3: // greater_than
					return d1 > compare ? 1 : 0;
				case 4: // less_or_equals_than
					return d1 <= compare ? 1 : 0;
				case 5: // greater_or_equals_than
					return d1 >= compare ? 1 : 0;
				case 6: // abs_less_than
					return nd4j::math::nd4j_abs<X>(d1) < compare ? 1 : 0;
				case 7: // abs_greater_than
					return nd4j::math::nd4j_abs<X>(d1) > compare ? 1 : 0;
				case 8: // is inf
					return nd4j::math::nd4j_isinf(d1) ? 1 : 0;
				case 9: // is nan
					return nd4j::math::nd4j_isnan(d1) ? 1 : 0;
				case 10:
					return (d1 == compare) ? 1 : 0;
				case 11:
					return (d1 != compare) ? 1 : 0;
				case 12: // abs_greater_or_equals_than
					return nd4j::math::nd4j_abs<X>(d1) >= compare ? 1 : 0;
				case 13: // abs_less_or_equals_than
					return nd4j::math::nd4j_abs<X>(d1) <= compare ? 1 : 0;
                case 14:
                    // isFinite
                    return !(nd4j::math::nd4j_isinf(d1) || nd4j::math::nd4j_isnan(d1)) ? 1 : 0;
                case 15:
                    // isInfinite
                    return nd4j::math::nd4j_isinf(d1) || nd4j::math::nd4j_isnan(d1) ? 1 : 0;
				default:
					printf("Undefined match condition: [%i]\n", mode);
			}

            return d1;
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
    };

	template <typename X>
	class ELU {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_elu<X,X>(d1);
		}
	};


	template <typename X>
	class ELUDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_eluderivative<X,X>(d1);
		}
	};


	template <typename X, typename Y, typename Z>
	class RELU {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			auto xt = static_cast<Z>(d1);
			auto xf = static_cast<Z>(d2);
			return xt < xf ? xf : xt;
		}
	};

    template <typename X, typename Y, typename Z>
    class SXELogitsSmoother {
    public:
        op_def static Z op(X d1, Y d2, Z *params) {
            return d1 * ((X)1.f - (X) d2) + (X)(0.5f) * (X) d2;
        }
    };

	template <typename X, typename Y, typename Z>
	class RELU6 {
	public:
	    no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			auto relu = simdOps::RELU<X,Y,Z>::op(d1, d2, params);
			return relu < static_cast<Z>(6) ? relu : static_cast<Z>(6);
		}
	};

	template <typename X, typename Y, typename Z>
	class LeakyRELU {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
		    auto val = static_cast<Z>(d1);
		    auto alpha = static_cast<Z>(d2);
		    return val < 0.0f ? alpha * val : val;
		}
	};

    template <typename X>
    class SELU {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 > static_cast<X>(0.0f) ? static_cast<X>(SELU_LAMBDA) * static_cast<X>(d1) : static_cast<X>(SELU_LAMBDA) * (static_cast<X>(SELU_ALPHA) * nd4j::math::nd4j_exp<X, X>(d1) - static_cast<X>(SELU_ALPHA));
        }
    };

    template <typename X>
    class SELUDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return d1 > static_cast<X>(0.f) ? static_cast<X>(SELU_LAMBDA) : static_cast<X>(SELU_ALPHA) * static_cast<X>(SELU_LAMBDA) * nd4j::math::nd4j_exp<X, X>(d1);
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


	template <typename X>
	class ASin {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_asin<X,X>(d1);
		}
	};

	template <typename X>
	class Sinh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_sinh<X,X>(d1);
		}
	};

	template <typename X>
	class SinhDerivative {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_cosh<X, X>(d1);
		}
	};

	template <typename X>
	class Cosh {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_cosh<X,X>(d1);
		}
	};


	template <typename X>
	class Tan {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_tan<X,X>(d1);
		}
	};

    template <typename X>
    class TanDerivative {
    public:
        no_op_exec_special_same
        no_op_exec_special_same_cuda

        op_def static X op(X d1, X *params) {
            return  static_cast<X>(1.f) / nd4j::math::nd4j_pow<X, X, X>(nd4j::math::nd4j_cos<X, X>(d1), static_cast<X>(2.0f));
        }
    };

	template <typename X>
	class ATan {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return nd4j::math::nd4j_atan<X, X>(d1);
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
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return d1;
		}
	};


	template <typename X>
	class Stabilize {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			X k = params[0];
			if (d1 * k > static_cast<X>(- MIN_CUTFOFF))
				return static_cast<X>(- MIN_CUTFOFF) / k;
			else if (d1 * k < static_cast<X>(MIN_CUTFOFF))
				return static_cast<X>(MIN_CUTFOFF) / k;
			return d1;
		}
	};



	template <typename X, typename Y, typename Z>
	class Step {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static Z op(X d1, Y d2, Z *params) {
			return (d1 > static_cast<X>(d2) ? static_cast<Z>(1) : static_cast<Z>(0));
		}
	};



	template <typename X>
	class OneMinus {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		op_def static X op(X d1, X *params) {
			return static_cast<X>(1) - d1;
		}
	};

	template <typename X>
	class Sum {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

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

    template <typename X>
    class ReduceSameBenchmarkOp {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

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
            auto f1 = static_cast<float>(d1);
            return static_cast<X>(nd4j::math::nd4j_pow<float,float,float>(f1, 3)
                   + nd4j::math::nd4j_log<float,float>(f1) * nd4j::math::nd4j_sin<float,float>(f1)
                     / nd4j::math::nd4j_tanh<float,float>(static_cast<float>(M_E) * static_cast<float>(M_PI) * f1)
                     * nd4j::math::nd4j_sqrt<float,float>(static_cast<float>(M_PI) / f1)
                   - nd4j::math::nd4j_atan<float,float>(static_cast<float>(M_E) / f1));
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

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, Z *extraParams) {
            auto p = d1 * d1;
            return static_cast<Z>(p) * nd4j::math::nd4j_log<X, Z>(p);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
            return -reduction;
        }
    };


    template <typename X, typename Z>
    class LogEntropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, Z *extraParams) {
			return static_cast<Z>(d1) * nd4j::math::nd4j_log<X, Z>(d1);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			//entropy is -sum(p(x) * log(p(x))); log entropy is log of this
			return nd4j::math::nd4j_log<Z, Z>(-reduction);
        }
    };

    template <typename X, typename Z>
    class Entropy {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, Z *extraParams) {
            return static_cast<Z>(d1) * nd4j::math::nd4j_log<X, Z>(d1);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
            return static_cast<Z>(-reduction);		//entropy is -sum(p(x) * log(p(x)))
        }
    };


    template <typename X>
    class ASum {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::ASUM;

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
        no_op_exec_special_accumulation_long
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::ASUM;

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0);
        }

        op_def static Z merge(Z old, Z opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, X *extraParams) {
            return d1 == static_cast<X>(0.0f) ? static_cast<Z>(0.0f) : static_cast<Z>(1.0f);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, X *extraParams) {
            return reduction;
        }
    };


    template <typename X, typename Z>
    class CountZero {
    public:
        no_op_exec_special_accumulation_long
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0.0f);
        }

        op_def static Z merge(Z old, Z opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, X *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, X *extraParams) {
            return d1 == static_cast<X>(0) ? static_cast<X>(1) : static_cast<X>(0);
        }

        op_def static Z postProcess(X reduction, Nd4jLong n, X *extraParams) {
            return static_cast<Z>(reduction);
        }
    };

	template <typename X>
	class Prod {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::PRODUCT;

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

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static Z merge(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static Z update(X old, X opOutput, X *extraParams) {
			return opOutput + old;
		}

		op_def static Z op(X d1, X *extraParams) {
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

        const static functions::ReduceType reduceType = functions::ReduceType::PRODUCT;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(1);
        }

        op_def static Z merge(X old, X opOutput, X *extraParams) {
            return opOutput * old;
        }

        op_def static Z update(X old, X opOutput, X *extraParams) {
            return opOutput * old;
        }

        op_def static Z op(X d1, X *extraParams) {
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

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z op(X d1, Z *extraParams) {
			return d1;
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return reduction / (Z) n;
		}
	};

    template <typename X, typename Z>
    class ReduceFloatBenchmarkOp {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z update(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, Z *extraParams) {
            auto f1 = static_cast<float>(d1);
            return static_cast<Z>(nd4j::math::nd4j_pow<float,float,float>(f1, 3)
                    + nd4j::math::nd4j_log<float,float>(f1) * nd4j::math::nd4j_sin<float,float>(f1)
                    / nd4j::math::nd4j_tanh<float,float>(static_cast<float>(M_E) * static_cast<float>(M_PI) * f1)
                    * nd4j::math::nd4j_sqrt<float,float>(static_cast<float>(M_PI) / f1)
                    - nd4j::math::nd4j_atan<float,float>(static_cast<float>(M_E) / f1));
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
            return (Z) reduction / (Z) n;
        }
    };


    template <typename X, typename Z>
    class AMean {
    public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

        op_def static X startingValue(const X *input) {
            return static_cast<X>(0);
        }

        op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(opOutput) + nd4j::math::nd4j_abs<X>(old);
        }

        op_def static Z update(Z old, Z opOutput, Z *extraParams) {
            return opOutput + old;
        }

        op_def static Z op(X d1, Z *extraParams) {
            return nd4j::math::nd4j_abs<X>(d1);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
            return nd4j::math::nd4j_abs<Z>(reduction) / static_cast<Z>(n);
        }
    };

	template <typename X>
	class Max {
	public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::MAX;

		op_def static X startingValue(const X *input) {
			return -nd4j::DataTypeUtils::infOrMax<X>();
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
		op_def static Z op(X d1, Y d2, Z *params) {
			return op(d1, d2);
		}

		op_def static Z op(X d1, Y d2) {
			auto z1 = static_cast<Z>(d1);
			auto z2 = static_cast<Z>(d2);

			if (nd4j::math::nd4j_abs<Z>(z1) > nd4j::math::nd4j_abs<Z>(z2))
				return z1;
			else
				return z2;
		}
	};


	template <typename X, typename Y, typename Z>
	class AMinPairwise {
	public:
		op_def static Z op(X d1, Y d2, Z *params) {
            return op(d1, d2);
		}

		op_def static Z op(X d1, Y d2) {
			auto z1 = static_cast<Z>(d1);
			auto z2 = static_cast<Z>(d2);

			if (nd4j::math::nd4j_abs<Z>(z1) < nd4j::math::nd4j_abs<Z>(z2))
				return z1;
			else
				return z2;
		}
	};

	template <typename X, typename Y, typename Z>
	class MaxPairwise {
	public:
		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_max<Z>(static_cast<Z>(d1), static_cast<Z>(d2));
		}

		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_max<Z>(static_cast<Z>(d1), static_cast<Z>(d2));
		}
	};


	template <typename X, typename Y, typename Z>
	class MinPairwise {
	public:
		op_def static Z op(X d1, Y d2, Z *params) {
			return nd4j::math::nd4j_min<Z>(static_cast<Z>(d1), static_cast<Z>(d2));
		}

		op_def static Z op(X d1, Y d2) {
			return nd4j::math::nd4j_min<Z>(static_cast<Z>(d1), static_cast<Z>(d2));
		}
	};

    template <typename X>
    class AMax {
    public:
        no_op_exec_special_accumulation_same
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::AMAX;

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
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::AMIN;

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
        no_op_exec_special_accumulation_same_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::MIN;

        op_def static X startingValue(const X *input) {
            return nd4j::DataTypeUtils::infOrMax<X>();
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

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static Z op(X d1, Z *extraParams) {
			return static_cast<Z>(nd4j::math::nd4j_abs<X>(d1));
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return reduction;
		}
	};


	template <typename X, typename Z>
	class Norm2 {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_sqrt<Z, Z>(reduction);
		}

        op_def static Z op(X d1, Z *extraParams) {
            return static_cast<Z>(d1 * d1);
        }
    };

	template <typename X, typename Z>
	class SquaredNorm {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z op(X d1, Z *extraParams) {
			return static_cast<Z>(d1 * d1);
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return reduction;
		}
	};

	template <typename X, typename Z>
	class NormFrobenius {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}


		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z op(X d1, Z *extraParams) {
			X v = nd4j::math::nd4j_abs<X>(d1);
			return static_cast<Z>(v * v);
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_sqrt<Z, Z>(reduction);
		}
	};

	template <typename X, typename Z>
	class NormP {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;
		}

		op_def static Z op(X d1, Z *extraParams) {
			return nd4j::math::nd4j_pow<X, Z, Z>(nd4j::math::nd4j_abs<X>(d1), extraParams[0]);
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_pow<Z, Z, Z>(reduction, static_cast<Z>(1.0f) / extraParams[0]);
		}
	};

	template <typename X, typename Z>
	class NormMax {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0);
		}

		op_def static Z merge(Z old, Z opOutput, Z *extraParams) {
			return opOutput + old;

		}

		op_def static Z update(Z old, Z opOutput, Z *extraParams) {
			return nd4j::math::nd4j_max<Z>(nd4j::math::nd4j_abs<Z>(old),
				nd4j::math::nd4j_abs<Z>(opOutput));
		}

		op_def static Z op(X d1, Z *extraParams) {
			return static_cast<Z>(d1);
		}

		op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParams) {
			return nd4j::math::nd4j_max<Z>(nd4j::math::nd4j_abs<Z>(reduction), nd4j::math::nd4j_abs<Z>(reduction));
		}
	};

	template <typename X, typename Z>
	class Variance {
	public:
        no_op_exec_special_accumulation
        no_op_exec_special_accumulation_cuda

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static Z merge(X old, X opOutput, Z *extraParams) {
			return old + opOutput;
		}

		op_def static Z update(X old, X opOutput, Z *extraParams) {
			return old + opOutput;

		}

		op_def static X op(X d1, Z *extraParams) {
			X mean = static_cast<X>(extraParams[0]);
			X ret = d1 - mean;
			return ret * ret;
		}

		op_def static Z postProcess(X reduction, Nd4jLong n, Z *extraParams) {
			// T bias = extraParams[1];
			// return (reduction - (nd4j::math::nd4j_pow<T>(bias, static_cast<T>(2.0f)) / static_cast<T>(n))) / (n - 1)
			return static_cast<Z>(reduction) / static_cast<Z>(n - 1);
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

        const static functions::ReduceType reduceType = functions::ReduceType::SUM;

		op_def static X startingValue(const X *input) {
			return static_cast<X>(0.0f);
		}

		op_def static Z merge(X old, X opOutput, Z *extraParams) {
			return old + opOutput;
		}

		op_def static Z update(X old, X opOutput, Z *extraParams) {
			return old + opOutput;

		}

		op_def static Z op(X d1, Z *extraParams) {
			X mean = extraParams[0];
			X ret = d1 - mean;
			return ret * ret;
		}

		op_def static Z postProcess(X reduction, Nd4jLong n, Z *extraParams) {
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

		op_def static Y startingValue(const X *input) {
			return static_cast<Y>(0.0f);
		}

		op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParams) {
			return reduction / (nd4j::math::nd4j_sqrt<Y, Y>(extraParams[0]) * nd4j::math::nd4j_sqrt<Y, Y>(extraParams[1]));
		}

		op_def static Y op(X d1, X d2, Y *extraParams) {
			extraParams[0] += static_cast<Y>(d1 * d1);
			extraParams[1] += static_cast<Y>(d2 * d2);
			return static_cast<Y>(d1 * d2);
		}

		op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
			extraParamsTotal[0] += extraParamsLocal[0];
			extraParamsTotal[1] += extraParamsLocal[1];
		}

#ifdef __CUDACC__
		static _CUDA_D inline Y opAtomic(X d1, X d2, Y *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],static_cast<Y>(d1 * d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1],static_cast<Y>(d2 * d2));

			return static_cast<Y>(d1 * d2);
		}
#endif

		op_def static Y update(Y old, Y opOutput, Y *extraParams) {
			return old + opOutput;
		}


		op_def static Y merge(Y old, Y opOutput, Y *extraParams) {
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

        op_def static Y startingValue(const X *input) {
            return static_cast<X>(0.0f);
        }

        op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParams) {
            // num / denom
            return (static_cast<Y>(1.0f)) - (extraParams[0] / extraParams[1]);
        }

        op_def static Y num(X d1, X d2) {
            return nd4j::math::nd4j_min<X>(d1, d2);
        }

        op_def static Y denom(X d1, X d2) {
            return nd4j::math::nd4j_max<X>(d1, d2);
        }

        op_def static Y op(X d1, X d2, Y *extraParams) {
            extraParams[0] += static_cast<Y>(num(d1, d2));
            extraParams[1] += static_cast<Y>(denom(d1, d2));
            return static_cast<Y>(0.0f);
        }

        op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
        __device__
		static inline Y opAtomic(X d1, X d2, Y *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0],num(d1, d2));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], denom(d1, d2));

			return static_cast<Y>(0.0f);
		}
#endif

        op_def static  Y update(Y old, Y opOutput, Y *extraParams) {
            return old + opOutput;
        }


        op_def static Y merge(Y old, Y opOutput, Y *extraParams) {
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

        op_def static Y startingValue(const X *input) {
            return static_cast<Y>(0.0f);
        }

        op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParams) {
            return static_cast<Y>(reduction / n);
        }

        op_def static Y op(X d1, X d2, Y *extraParams) {
            return (d1 == d2) ? static_cast<Y>(0.0f) :  static_cast<Y>(1.0f);
        }

        op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {

        }

#ifdef __CUDACC__
        __device__
		static inline Y opAtomic(X d1, X d2, Y *extraParams) {
			return op(d1, d2, extraParams);
		}
#endif

        op_def static Y update(Y old, Y opOutput, Y *extraParams) {
            return old + opOutput;
        }


        op_def static Y merge(Y old, Y opOutput, Y *extraParams) {
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

        op_def static Y startingValue(const X *input) {
            return static_cast<Y>(0.0f);
        }

        op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParams) {
            return (static_cast<Y>(1.0f)) - (reduction / (nd4j::math::nd4j_sqrt<Y, Y>(extraParams[0]) * nd4j::math::nd4j_sqrt<Y, Y>(extraParams[1])));
        }

        op_def static Y op(X d1, X d2, Y *extraParams) {
            extraParams[0] += static_cast<Y>(nd4j::math::nd4j_abs<X>(d1) * nd4j::math::nd4j_abs<X>(d1));
            extraParams[1] += static_cast<Y>(nd4j::math::nd4j_abs<X>(d2) * nd4j::math::nd4j_abs<X>(d2));
            return (d1 * d2);
        }

        op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
            extraParamsTotal[0] += extraParamsLocal[0];
            extraParamsTotal[1] += extraParamsLocal[1];
        }

#ifdef __CUDACC__
	static _CUDA_D inline Y opAtomic(X d1, X d2, Y *extraParams) {
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[0], nd4j::math::nd4j_abs<Y>(d1) * nd4j::math::nd4j_abs<Y>(d1));
			nd4j::math::atomics::nd4j_atomicAdd(&extraParams[1], nd4j::math::nd4j_abs<Y>(d2) * nd4j::math::nd4j_abs<Y>(d2));

			return (d1 * d2);
		}
#endif

        op_def static Y update(Y old, Y opOutput, Y *extraParams) {
            return old + opOutput;
        }


        op_def static Y merge(Y old, Y opOutput, Y *extraParams) {
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

		op_def static Y startingValue(const X *input) {
			return static_cast<Y>(0.0f);
		}

		op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParamsRef) {
			return reduction;
		}

		op_def static Y op(X d1, X d2, Y *extraParamsRef) {
			return static_cast<Y>(d1 * d2);
		}


#ifdef __CUDACC__
		__device__
		static inline Y opAtomic(X d1, X d2, Y *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static Y update(Y old, Y opOutput, Y *extraParamsRef) {
			return opOutput + old;
		}

		op_def static Y merge(Y old, Y opOutput, Y *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}

		op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}
	};


    /**
	* Op to check equality within arrays
	*/
    template <typename X, typename Z>
    class EqualsWithEps {
    public:
        static const int extraParamsLen = 0;

        op_def static X * generateExtraParams() {
            return nullptr;
        }

        op_def static void finalizeExtraParams(X *extraParamsRef) {
            //no-op
        }

        op_def static Z startingValue(const X *input) {
            return static_cast<Z>(0.0f);
        }

        op_def static Z postProcess(Z reduction, Nd4jLong n, Z *extraParamsRef) {
            return reduction;
        }

        op_def static Z op(X d1, X d2, Z *extraParamsRef) {        	
			double eps = nd4j::math::nd4j_abs<double>(extraParamsRef[2]);
			return static_cast<Z>(!nd4j::math::nd4j_eq<X>(d1, d2, eps));
        }


#ifdef __CUDACC__
        __device__
		static inline Z opAtomic(X d1, X d2, Z *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

        op_def static Z update(Z old, Z opOutput, Z *extraParamsRef) {
            return opOutput + old;
        }

        op_def static Z merge(X old, Z opOutput, Z *extraParamsRef) {
            return update(old, opOutput, extraParamsRef);
        }

        op_def static void aggregateExtraParams(Z *extraParamsTotal, Z *extraParamsLocal) {}
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

		op_def static Y startingValue(const X *input) {
			return static_cast<Y>(0.0f);
		}

		op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParamsRef) {
			return nd4j::math::nd4j_sqrt<Y, Y>(reduction);
		}

		op_def static Y op(X d1, X d2, Y *extraParamsRef) {
			X ret = d1 - d2;
			return static_cast<Y>(ret * ret);
		}


#ifdef __CUDACC__
			__device__
			static  inline Y opAtomic(X d1, X d2, Y *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

		op_def static Y update(Y old, Y opOutput, Y *extraParamsRef) {
			return opOutput + old;
		}

		op_def static Y merge(Y old, Y opOutput, Y *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
		op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}

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

		op_def static Y startingValue(const X *input) {
			return static_cast<Y>(0.0f);
		}

		op_def static Y postProcess(Y reduction, Nd4jLong n, Y *extraParamsRef) {
			return reduction;
		}

		op_def static Y op(X d1, X d2, Y *extraParamsRef) {
			return nd4j::math::nd4j_abs<X>(d1 - d2);
		}

		op_def static Y update(Y old, Y opOutput, Y *extraParamsRef) {
			return old + opOutput;
		}

		op_def static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {

		}


#ifdef __CUDACC__
		__device__
		static inline Y opAtomic(X d1, X d2, Y *extraParamsRef) {
			return op(d1, d2, extraParamsRef);
		}
#endif

#ifndef __clang__
#pragma omp declare simd uniform(extraParamsRef)
#endif
		op_def static Y merge(X old, X opOutput, X *extraParamsRef) {
			return update(old, opOutput, extraParamsRef);
		}
	};


	template <typename X>
	class IndexAbsoluteMax  {
	public:
		static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return nd4j::math::nd4j_abs<X>(val);
		}

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
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

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
		functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (nd4j::math::nd4j_abs<X>(f1.value) > nd4j::math::nd4j_abs<X>(f2.value))
				return f2;
			return f1;
		}

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

		static _CUDA_HD inline X startingValue(const X *input) {
			return 0;
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

		static _CUDA_HD  inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
		functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};

    template <typename X>
    class FirstIndex {
    public:
        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
            return val;
        }

        static _CUDA_HD  functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {

#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            auto res = simdOps::MatchCondition<X,X>::op(opOutput.value, extraParams);

			//printf("res: %f; oldIdx: %i; newIdx: %i\n", res, old.index, opOutput.index);

            if (res == static_cast<X>(0))
                return old;

            if (old.index < 0)
                return opOutput;

            if (old.index > opOutput.index)
                return opOutput;

            return old;
        }

        static _CUDA_HD inline X startingValue(const X *input) {
            return -nd4j::DataTypeUtils::infOrMax<X>();
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                               functions::indexreduce::IndexValue<X> d2, X *extraParams) {
            return d1;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
                functions::indexreduce::IndexValue<X> f1,
                functions::indexreduce::IndexValue<X> f2, X *extraParams) {
            if (f1.index > f2.index)
                return f2;
            return f1;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
                functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
                X *dx, int incx, X *extraParams, X *result) {
            return reduction;
        }
    };


    template <typename X>
    class LastIndex {
    public:
        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
            return val;
        }

        static _CUDA_HD functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
#ifdef __CUDACC__
            if (opOutput.index < 0)
                return old;
#endif

            auto res = simdOps::MatchCondition<X,X>::op(opOutput.value, extraParams);

            if (res == static_cast<X>(0))
                return old;

            if (old.index < 0)
                return opOutput;

            if (old.index < opOutput.index)
                return opOutput;

            return old;
        }

        static _CUDA_HD inline X startingValue(const X *input) {
            return -nd4j::DataTypeUtils::infOrMax<X>();
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = -1;
            return local;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                               functions::indexreduce::IndexValue<X> d2, X *extraParams) {
            return d1;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
                functions::indexreduce::IndexValue<X> f1,
                functions::indexreduce::IndexValue<X> f2, X *extraParams) {
            if (f1.index < f2.index)
                return f2;
            return f1;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
                functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
                X *dx, int incx, X *extraParams, X *result) {
            return reduction;
        }
    };


	template <typename X>
	class IndexMax  {
	public:

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

        static _CUDA_HD functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
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

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
				functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (f1.value > f2.value)
				return f2;
			return f1;
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

        static _CUDA_HD inline X startingValue(const X *input) {
			return -nd4j::DataTypeUtils::infOrMax<X>();
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
				functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};


	template <typename X>
	class IndexAbsoluteMin {
	public:
		static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(
				functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

		static _CUDA_HD inline X startingValue(const X *input) {
			return nd4j::DataTypeUtils::infOrMax<X>();
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
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

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
		functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (nd4j::math::nd4j_abs<X>(f1.value) < nd4j::math::nd4j_abs<X>(f2.value))
				return f2;
			return f1;
		}

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

		static _CUDA_HD  inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
		functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};


	template <typename X>
	class IndexMin {
	public:
        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(
				functions::indexreduce::IndexValue<X> val, X *extraParams) {
			return val;
		}

        static _CUDA_HD inline X startingValue(const X *input) {
			return nd4j::DataTypeUtils::infOrMax<X>();
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> startingIndexValue(X *input) {
            functions::indexreduce::IndexValue<X> local;
            local.value = startingValue(input);
            local.index = 0;
            return local;
        }

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
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

		static _CUDA_HD inline functions::indexreduce::IndexValue<X> merge(
				functions::indexreduce::IndexValue<X> f1,
				functions::indexreduce::IndexValue<X> f2, X *extraParams) {
			if (f1.value < f2.value)
				return f2;
			return f1;
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> postProcess(
				functions::indexreduce::IndexValue<X> reduction, int n, int xOffset,
				X *dx, int incx, X *extraParams, X *result) {
			return reduction;
		}

        static _CUDA_HD inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
				functions::indexreduce::IndexValue<X> d2, X *extraParams) {
			return d1;
		}
	};

	template <typename X, typename Z>
	class SummaryStatsVariance {
	public:

        static _CUDA_HD inline Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
			if (biasCorrected) {
				Z ret = static_cast<Z>(val.varianceBiasCorrected());
				if (ret < static_cast<Z>(0.0f))
					return static_cast<Z>(val.variance());
				return ret;
			}
			return static_cast<Z>(val.variance());
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<X> op(functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
			return d1;
		}
	};

	template <typename X, typename Z>
	class SummaryStatsStandardDeviation {
	public:

        static _CUDA_HD inline Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
			if (biasCorrected) {
				auto ret = static_cast<Z>(val.varianceBiasCorrected());
				if (ret < static_cast<Z>(0.0f))
					return nd4j::math::nd4j_sqrt<double, Z>(val.variance());
				else
					return nd4j::math::nd4j_sqrt<double, Z>(ret);
			}
			return  nd4j::math::nd4j_sqrt<double, Z>(val.variance());
		}

        static _CUDA_HD inline functions::summarystats::SummaryStatsData<X> op(functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
			return d1;
		}
	};

    template <typename X>
	class DropOut {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda

		inline _CUDA_D static X op(X d1, X *params) {
			X prob = params[0];

#ifdef __CUDACC__
			X length = params[1];
            X tid = blockIdx.x * blockDim.x + threadIdx.x;
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
			X tid = blockIdx.x * blockDim.x + threadIdx.x;
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
        	auto zd1 = static_cast<Z>(d1);
			auto zd2 = static_cast<Z>(d2);
			auto compare = params[0];
            auto eps = params[2];
            int mode = (int) params[3];
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<Z>(zd1 - compare) <= eps)
                    return zd2;
                else
                    return zd1;
            else if (mode == 1) // not equals eps
                if (nd4j::math::nd4j_abs<Z>(zd1 - compare) > eps)
                    return zd2;
                else
                    return zd1;
            else if (mode == 2) // less_than eps
                if (zd1 < compare)
                    return zd2;
                else
                    return zd1;
            else if (mode ==3) // greater_than
                if (zd1 > compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 4) // less_or_equals_than
                if (zd1 <= compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 5) // greater_or_equals_than
                if (zd1 >= compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 6) // abs_less_than
                if (nd4j::math::nd4j_abs<Z>(zd1) < compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<Z>(zd1) > compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 8) // is inf
                if (nd4j::math::nd4j_isinf(zd1))
                    return zd2;
                else
                    return zd1;
            else if (mode == 9) // is nan
                if (nd4j::math::nd4j_isnan(zd1))
                    return zd2;
                else
                    return zd1;
            else if (mode == 10)
                if (zd1 == compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 11)
                if (zd1 != compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 12) // abs_greater_or_equals_than
                if (nd4j::math::nd4j_abs<Z>(zd1) >= compare)
                    return zd2;
                else
                    return zd1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<Z>(zd1) <= compare)
                    return zd2;
                else
                    return zd1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return zd1;
        }
    };

	template <typename X, typename Y, typename Z>
	class CompareAndSet {
	public:


        // op definition for PairWise Transform
        op_def static Z op(X dX, Y dY, Z *params) {
			auto d1 = static_cast<Z>(dX);
		    auto d2 = static_cast<Z>(dY);
            auto compare = params[0];
			auto eps = params[2];
			auto mode = static_cast<int>(params[3]);
            if (mode == 0) // equals
                if (nd4j::math::nd4j_abs<Z>(d2 - compare) <= eps)
                    return d2;
                else
                    return d1;
            else if (mode == 1) // not equals
                if (nd4j::math::nd4j_abs<Z>(d2 - compare) > eps)
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
                if (nd4j::math::nd4j_abs<Z>(d2) < compare)
                    return d2;
                else
                    return d1;
            else if (mode == 7) // abs_greater_than
                if (nd4j::math::nd4j_abs<Z>(d2) > compare)
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
                if (nd4j::math::nd4j_abs<Z>(d1) >= compare)
                    return d2;
                else
                    return d1;
            else if (mode == 13) // abs_less_or_equals_than
                if (nd4j::math::nd4j_abs<Z>(d1) <= compare)
                    return d2;
                else
                    return d1;
            else
                printf("Undefined boolean operation: [%i]\n", mode);
            return d1;
        }
	};

	template <typename X>
	class CompareAndSetTransform {
	public:
		no_op_exec_special_same
		no_op_exec_special_same_cuda


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
	};


}

#endif
	
