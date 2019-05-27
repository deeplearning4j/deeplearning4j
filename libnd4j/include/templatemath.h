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

/*
 * templatemath.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TEMPLATEMATH_H_
#define TEMPLATEMATH_H_

#include <dll.h>
#include <pointercast.h>
#include <platformmath.h>


#define BFLOAT16_MAX_VALUE 32737.
#define HALF_MAX_VALUE 65504.
#define FLOAT_MAX_VALUE 3.4028235E38
#define DOUBLE_MAX_VALUE 1.7976931348623157E308
#define FLOAT_MIN_NORMAL 1.17549435e-38

#ifndef M_E
#define M_E 2.718281828459
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace nd4j {
#ifdef __CUDACC__

#endif

	namespace math {
		template<typename T>
		math_def inline T nd4j_abs(T value);

        template<typename T>
        math_def inline void nd4j_swap(T &val1, T &val2);

		template<typename T>
        math_def inline T nd4j_max(T val1, T val2);

		template<typename T>
        math_def inline T nd4j_min(T val1, T val2);

		template <typename T>
		math_def inline bool nd4j_eq(T val1, T val2, double eps);

		template<typename T, typename Z>
		math_def inline Z nd4j_re(T val1, T val2);

		template<typename T, typename Z>
        math_def inline Z nd4j_rint(T val1);

		template<typename T, typename Z>
		math_def inline Z nd4j_copysign(T val1, T val2);

//#ifndef __CUDACC__
        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_dot(X *x, Y *y, int length);
//#endif

		template<typename T, typename Z>
        math_def inline Z nd4j_ceil(T val1);

		template<typename T>
        math_def inline bool nd4j_isnan(T val1);

		template<typename T>
        math_def inline bool nd4j_isinf(T val1);

		template<typename T>
        math_def inline bool nd4j_isfin(T val1);

		template<typename T, typename Z>
        math_def inline Z nd4j_cos(T val);

        template<typename T, typename Z>
        math_def inline Z nd4j_cosh(T val);

		template<typename X, typename Z>
        math_def inline Z nd4j_exp(X val);

		template<typename T, typename Z>
        math_def inline Z nd4j_floor(T val);

		template<typename X, typename Z>
        math_def inline Z nd4j_log(X val);

		template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_pow(X val, Y val2);

		template<typename T, typename Z>
        math_def inline Z nd4j_round(T val);

        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_remainder(X num, Y denom);

        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_fmod(X num, Y denom);

		template<typename T, typename Z>
        math_def inline Z nd4j_erf(T num);

		template<typename T, typename Z>
        math_def inline Z nd4j_erfc(T num);

		template<typename T, typename Z>
        math_def inline Z nd4j_sigmoid(T val) {
			return (Z) 1.0f / ((Z) 1.0f + nd4j_exp<T, Z>(-val));
		}

		template<typename T, typename Z>
        math_def inline Z nd4j_elu(T val) {
			if (val >= (T) 0.f) return val;
			else return nd4j_exp<T, Z>(val) - (Z) 1.0f;
			//return val >= 0.0 ? val : (nd4j_exp<T>(val) - 1.0);
		}


		template<typename T, typename Z>
        math_def inline Z nd4j_leakyrelu(T val,T alpha) {
			if (val < (T) 0.0f)
			    return alpha * val;
			else
			    return val;
		}


		template<typename T, typename Z>
        math_def inline Z nd4j_eluderivative(T val) {
			if (val >= (T) 0.0f) return (Z) 1.0f;
			else return nd4j_exp<T, Z>(val);
			//return val >= 0.0 ? 1.0 : nd4j_exp(val);
		}
		template<typename T, typename Z>
        math_def inline Z nd4j_sin(T val);

		template<typename T, typename Z>
		math_def inline Z nd4j_sinh(T val);

		template<typename T, typename Z>
        math_def inline Z softplus(T val) {
			return nd4j_log<T, Z>((Z) 1.0f + nd4j_exp<T, Z>(val));
		}

		template<typename T, typename Z>
        math_def inline Z nd4j_softsign(T val) {
			return val / ((T) 1.0f + nd4j::math::nd4j_abs<T>(val));
		}

		template<typename X, typename Z>
        math_def inline Z nd4j_sqrt(X val);

		template<typename X, typename Z>
        math_def inline Z nd4j_tanh(X val);

        template<typename T, typename Z>
        math_def inline Z nd4j_tan(T val);

		template<typename X, typename Z>
		math_def inline Z nd4j_atan2(X val1, X val2);

		template<typename X, typename Z>
		math_def inline Z nd4j_atan2(X val1, X val2) {
            return p_atan2<Z>(static_cast<Z>(val1), static_cast<Z>(val2));
		}


        template<typename T, typename Z>
        math_def inline Z nd4j_tan(T tval) {
            return p_tan<Z>(static_cast<Z>(tval));
        }

        template<typename T, typename Z>
        math_def inline Z nd4j_tanhderivative(T val) {
			Z tanh = nd4j_tanh<T,Z>(val);
			return (Z) 1.0f - tanh * tanh;
		}
		template <typename T, typename Z>
        math_def inline T nd4j_sigmoidderivative(T val) {
			Z sigmoid = nd4j_sigmoid<T,Z>(val);
			return sigmoid * ((Z) 1.0f - sigmoid);
		}

		template<typename T, typename Z>
        math_def inline T nd4j_softsignderivative(T val) {
			T y = (T) 1.0f + nd4j_abs(val);
			return (Z) 1.0f / (y * y);
		}

        template<typename T, typename Z>
        math_def inline T nd4j_sgn(T val) {
            return val < (T) 0.0f ? (Z) -1.0f : val > (T) 0.0f ? (Z) 1.0f : (Z) 0.0f;
        }

        template<typename T, typename Z>
        math_def inline Z nd4j_sign(T val) {
            return nd4j_sgn<T, Z>(val);
        }

        template<typename T, typename Z>
        math_def inline Z nd4j_signum(T val) {
            return nd4j_sgn<T, Z>(val);
        }

//#ifndef __CUDACC__
/*
        template<>
        math_def inline float16 nd4j_dot<float16>(float16 *x, float16 *y, int length) {
            float16 dot = (float16) 0.0f;

            // TODO: since we can't use simd on unions, we might use something else here.
            for(int e = 0; e < length; e++) {
                dot += x[e] * y[e];
            }

            return dot;
        }
        */

		template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_dot(X *x, Y *y, int length) {
            Z dot = (Z)0.0f;

			for(int e = 0; e < length; e++) {
				dot += static_cast<Z>(x[e]) * static_cast<Z>(y[e]);
			}

			return dot;
		}
//#endif

		template<typename T, typename Z>
        math_def inline Z nd4j_acos(T val);

        template<typename T, typename Z>
        math_def inline Z nd4j_sech(T val);

		template<typename T, typename Z>
		math_def inline Z nd4j_acosh(T val);

		template<typename T, typename Z>
        math_def inline Z nd4j_asin(T val);

		template<typename T, typename Z>
		math_def inline Z nd4j_asinh(T val);

        template<typename T, typename Z>
        math_def inline Z nd4j_asinh(T val) {
            //Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x)
            return nd4j_log<Z, Z>(nd4j_sqrt<Z, Z>(nd4j_pow<T,T,Z>(val, (T) 2) + (Z) 1.f) + (Z) val);
        }

		template<typename T, typename Z>
        math_def inline Z nd4j_atan(T val);

        template<typename T, typename Z>
        math_def inline Z nd4j_atanh(T val);


        template<>
        math_def inline float16 nd4j_abs<float16>(float16 value) {
#ifdef NATIVE_HALFS
			if (value < (float16) 0.f) {
				 return float16(__hneg(value.data));
			} else 
				return value;
#else
			return (float16) fabsf((float) value);
#endif
		}
        template<>
        math_def inline bfloat16 nd4j_abs<bfloat16>(bfloat16 value) {
		return (bfloat16) fabsf((float) value);
        }
		template<>
        math_def inline float nd4j_abs<float>(float value) {
			return fabsf(value);
		}

		template<>
        math_def inline double nd4j_abs<double>(double value) {
			return fabs(value);
		}

		template<>
        math_def inline int nd4j_abs<int>(int value) {
			return abs(value);
		}

		template<>
		math_def inline Nd4jLong nd4j_abs<Nd4jLong>(Nd4jLong value) {
			return llabs(value);
		}

		template<>
		math_def inline bool nd4j_abs<bool>(bool value) {
			return value;
		}

		template<>
		math_def inline uint8_t nd4j_abs<uint8_t>(uint8_t value) {
			return value;
		}

		template<>
		math_def inline uint16_t nd4j_abs<uint16_t>(uint16_t value) {
			return value;
		}

		template<>
		math_def inline uint32_t nd4j_abs<uint32_t>(uint32_t value) {
			return value;
		}

		template<>
		math_def inline Nd4jULong nd4j_abs<Nd4jULong>(Nd4jULong value) {
			return value;
		}

		template<>
		math_def inline int8_t nd4j_abs<int8_t>(int8_t value) {
			return value < 0 ? -value : value;
		}

		template<>
		math_def inline int16_t nd4j_abs<int16_t>(int16_t value) {
			return value < 0 ? -value : value;
		}


		template<>
        math_def inline bool nd4j_isnan<float16>(float16 value) {
			return *(value.data.getXP()) == 0x7fffU;
		}

		template<>
		math_def inline bool nd4j_isnan<bfloat16>(bfloat16 value) {
			return value == bfloat16::nan(); //0x7fffU;
		}

		template<>
        math_def inline bool nd4j_isnan<float>(float value) {
			return value != value;
		}

		template<>
        math_def inline bool nd4j_isnan<double>(double value) {
			return value != value;
		}

		template<>
		math_def inline bool nd4j_isnan<int>(int value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<uint32_t>(uint32_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<uint16_t>(uint16_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<uint8_t>(uint8_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<int16_t>(int16_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<int8_t>(int8_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<bool>(bool value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<Nd4jLong>(Nd4jLong value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isnan<Nd4jULong>(Nd4jULong value) {
			return false;
		}

		template<>
        math_def inline bool nd4j_isinf<float16>(float16 value) {
			return value < (float16) -HALF_MAX_VALUE || value > (float16) HALF_MAX_VALUE;
		}

		template<>
		math_def inline bool nd4j_isinf<bfloat16>(bfloat16 value) {
			return value < (bfloat16) -BFLOAT16_MAX_VALUE || value > (bfloat16) BFLOAT16_MAX_VALUE;
		}

		template<>
        math_def inline bool nd4j_isinf<float>(float value) {
#ifdef __CUDACC__
            return isinf(value);
#else
            return std::isinf(value);
#endif
            //return value < -FLOAT_MAX_VALUE || value > FLOAT_MAX_VALUE;
		}

		template<>
        math_def inline bool nd4j_isinf<double>(double value) {
#ifdef __CUDACC__
            return isinf(value);
#else
            return std::isinf(value);
#endif
            //return value < -DOUBLE_MAX_VALUE || value > DOUBLE_MAX_VALUE;
		}

		template<>
        math_def inline bool nd4j_isinf<int>(int value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<uint32_t>(uint32_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<uint16_t>(uint16_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<uint8_t>(uint8_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<int16_t>(int16_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<int8_t>(int8_t value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<bool>(bool value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<Nd4jLong>(Nd4jLong value) {
			return false;
		}

		template<>
		math_def inline bool nd4j_isinf<Nd4jULong>(Nd4jULong value) {
			return false;
		}

		template<typename T>
        math_def inline bool nd4j_isfin(T value) {
			return !nd4j_isnan<T>(value) && !nd4j_isinf<T>(value);
		}

		template<>
		math_def inline float16 nd4j_copysign<float16>(float16 val1, float16 val2) {
			return (float16) copysignf((float) val1, (float) val2);
		}

		template<>
		math_def inline float nd4j_copysign<float>(float val1, float val2) {
			return copysignf(val1, val2);
		}

		template<>
		math_def inline double nd4j_copysign<double>(double val1, double val2) {
			return copysign(val1, val2);
		}

		template<>
		math_def inline int nd4j_copysign<int>(int val1, int val2) {
			if (val2 < 0) return -(nd4j_abs<int>(val1));
			else return nd4j_abs<int>(val1);
		}

		template<>
		math_def inline Nd4jLong nd4j_copysign<Nd4jLong>(Nd4jLong val1, Nd4jLong val2) {
			if (val2 < 0) return -(nd4j_abs<Nd4jLong>(val1));
			else return nd4j_abs<Nd4jLong>(val1);
		}

		template<>
		math_def inline bool nd4j_max(bool val1, bool val2) {
			return (val1 || val2) ? true : false;
		}

		template<typename T>
		math_def inline T nd4j_max(T val1, T val2) {
			return val1 > val2 ? val1 : val2;
		}

		template<>
		math_def inline bool nd4j_min(bool val1, bool val2) {
			return (val1 && val2) ? true : false;
		}

		template<typename T>
		math_def inline T nd4j_min(T val1, T val2) {
			return val1 < val2 ? val1 : val2;
		}

		template <typename T>
		math_def inline bool nd4j_eq(T d1, T d2, double eps) {
			if (nd4j::math::nd4j_isinf<T>(d1) && nd4j::math::nd4j_isinf<T>(d2)) {
				if (d1 > 0 && d2 > 0)
					return true;
				else if (d1 < 0 && d2 < 0)
					return true;
				else
					return false;
			}

			auto diff = static_cast<double>(nd4j::math::nd4j_abs<T>(d1 - d2));


			// works well except in the range of very large numbers
			if (diff <= eps)
				return true;

			// Knuth approach
			// works well except in the range of very small numbers
			if (diff <= nd4j::math::nd4j_max<double>(nd4j::math::nd4j_abs<double>(static_cast<double>(d1)), nd4j::math::nd4j_abs<double>(static_cast<double>(d2))) * eps)
				return true;

			return false;
		}

		template <typename X, typename Z>
        math_def inline Z nd4j_ceil(X val) {
            return static_cast<Z>(p_ceil<X>(val));
		}

        template <typename X, typename Z>
        math_def inline Z nd4j_round(X val) {
            return static_cast<Z>(p_round<X>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_asin(X val) {
            return p_asin<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_atan(X val) {
            return p_atan<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_atanh(X val) {
            return p_atanh<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_cosh(X val) {
            return p_cosh<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_rint(X val) {
            return p_rint<X>(val);
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_sinh(X val) {
            return p_sinh<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_acos(X val) {
            return p_acos<Z>(static_cast<Z>(val));
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_sech(X val) {
            return static_cast<Z>(1) / nd4j_cosh<X,Z>(val);
        }

        template <typename X, typename Z>
        math_def inline Z nd4j_acosh(X val) {
            return p_acosh<Z>(static_cast<Z>(val));
        }

		template <typename X, typename Z>
        math_def inline Z nd4j_cos(X val) {
            return p_cos<Z>(static_cast<Z>(val));
		}

		template <typename X, typename Z>
        math_def inline Z nd4j_exp(X val) {
            return p_exp<X>(val);
        }

		template<typename X, typename Z>
        math_def inline Z nd4j_floor(X val) {
            return static_cast<Z>(p_floor<X>(val));
		}

		template<typename X, typename Z>
        math_def inline Z nd4j_log(X val) {
            return static_cast<Z>(p_log<X>(val));
		}

		/**
		 * This func is special case - it must return floating point value, and optionally Y arg can be floating point argument
		 * @tparam X
		 * @tparam Y
		 * @tparam Z
		 * @param val
		 * @param val2
		 * @return
		 */
		template <typename X, typename Y, typename Z>
        math_def inline Z nd4j_pow(X val, Y val2) {
            return p_pow<Z>(static_cast<Z>(val), static_cast<Z>(val2));
		}



		template<typename T>
		math_def inline T nd4j_re(T val1, T val2) {
			if (val1 == (T) 0.0f && val2 == (T) 0.0f)
				return (T) 0.0f;

			return nd4j_abs<T>(val1 - val2) / (nd4j_abs<T>(val1) + nd4j_abs<T>(val2));
        }


        template <typename X, typename Y, typename Z>
		math_def inline Z nd4j_remainder(X val, Y val2) {
            return p_remainder<Z>(static_cast<Z>(val), static_cast<Z>(val2));
		}

		template <typename X, typename Y, typename Z>
		math_def inline Z nd4j_fmod(X val, Y val2) {
            return p_fmod<Z>(static_cast<Z>(val), static_cast<Z>(val2));
		}


		template <typename X, typename Z>
        math_def inline Z nd4j_sin(X val) {
            return p_sin<Z>(static_cast<Z>(val));
		}


		template <typename X, typename Z>
        math_def inline Z nd4j_sqrt(X val) {
            return p_sqrt<Z>(static_cast<Z>(val));
        }


        template <typename X>
        math_def inline X neg_tanh(X val) {
            X o = static_cast<X>(1.0f);
            X t = static_cast<X>(2.0f);
            X e = static_cast<X>(M_E);

            auto p = nd4j::math::nd4j_pow<X, X, X>(e, val * t);
            return (p - o)/ (p + o);
        }

        template <typename X>
        math_def inline X pos_tanh(X val) {
            X o = static_cast<X>(1.0f);
            X t = static_cast<X>(-2.0f);
            X e = static_cast<X>(M_E);

            auto p = nd4j::math::nd4j_pow<X, X, X>(e, val * t);
            return (o - p) / (o + p);
        }


		template <typename X, typename Z>
		math_def inline Z nd4j_tanh(X val) {
            return val <= 0 ? neg_tanh(val) : pos_tanh(val);
            //return p_tanh<Z>(static_cast<Z>(val));
		}

        template <typename X, typename Z>
        math_def inline Z nd4j_erf(X val) {
            return p_erf<Z>(static_cast<Z>(val));
        }


        template <typename X, typename Z>
        math_def inline Z nd4j_erfc(X val) {
            return p_erfc<Z>(static_cast<Z>(val));
        }

        template<typename T>
        math_def inline void nd4j_swap(T &val1, T &val2) {
            T temp = val1; val1=val2; val2=temp;
		};

#ifdef __CUDACC__
		namespace atomics {
template <typename T>
inline __device__ T nd4j_atomicAdd(T* address, T val);

template <typename T>
inline __device__ T nd4j_atomicSub(T* address, T val);
template <typename T>
inline __device__ T nd4j_atomicMul(T* address, T val);
template <typename T>
inline __device__ T nd4j_atomicDiv(T* address, T val);

template <typename T>
inline __device__ T nd4j_atomicMin(T* address, T val);
template <typename T>
inline __device__ T nd4j_atomicMax(T* address, T val);

template <>
inline __device__ int32_t nd4j_atomicMin<int32_t>(int32_t* address, int32_t val)  {
     return atomicMin(address, val);
}

template <>
inline __device__ uint32_t nd4j_atomicMin<uint32_t>(uint32_t* address, uint32_t val)  {
     return atomicMin(address, val);
}
template <>
inline __device__ float nd4j_atomicMin<float>(float* address, float val)  {
     return __int_as_float(atomicMin(reinterpret_cast<int*>(address), __float_as_int(val)));
}
template <>
inline __device__ double nd4j_atomicMin<double>(double* address, double val)  {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = __double_as_longlong(val);
    if (val >= 0 && *address >= 0)
        return __longlong_as_double(atomicMin(address_as_ull, old));
    else if (val < 0 && *address < 0)
        return __longlong_as_double(atomicMax(address_as_ull, old));
    else if (val < 0 && *address >= 0) {
        *address = val;
    }
    return *address;

}
template <>
inline __device__ unsigned long long nd4j_atomicMin<unsigned long long>(unsigned long long* address, unsigned long long val)  {
     return atomicMin(address, val);
}
template <>
inline __device__ Nd4jLong nd4j_atomicMin<Nd4jLong>(Nd4jLong* address, Nd4jLong val)  {
    if (val >= 0 && *address >= 0)
        return (Nd4jLong)atomicMin((unsigned long long*)address, (unsigned long long)val);
    else if (val < 0 && *address < 0)
        return (Nd4jLong)atomicMax((unsigned long long*)address, (unsigned long long)val);
    else if (val < 0 && *address >= 0)
        *address = val;
    return *address;
}

template <>
inline __device__ int16_t nd4j_atomicMin<int16_t>(int16_t* address, int16_t val)  {
    int32_t temp = *address;
    *address = atomicMin(&temp, (int)val);
    return *address;
}
template <>
inline __device__ bfloat16 nd4j_atomicMin<bfloat16>(bfloat16* address, bfloat16 val)  {
     return bfloat16(nd4j_atomicMin<int16_t>(&address->_data, val._data));
}
template <>
inline __device__ float16 nd4j_atomicMin<float16>(float16* address, float16 val)  {
     return float16(nd4j_atomicMin<int16_t>(reinterpret_cast<int16_t*>(&address->data), (int16_t)val.data));
}
template <>
inline __device__ int32_t nd4j_atomicMax<int32_t>(int32_t* address, int32_t val)  {
     return atomicMax(address, val);
}

template <>
inline __device__ uint8_t nd4j_atomicMin<uint8_t>(uint8_t* address, uint8_t val)  {
    uint32_t temp = *address;
    *address = atomicMin(&temp, (uint32_t)val);
    return *address;
}
template <>
inline __device__ int8_t nd4j_atomicMin<int8_t>(int8_t* address, int8_t val)  {
    int32_t temp = *address;
    *address = atomicMin(&temp, (int)val);
    return *address;
}
template <>
inline __device__ uint16_t nd4j_atomicMin<uint16_t>(uint16_t* address, uint16_t val)  {
    uint32_t temp = *address;
    *address = atomicMin(&temp, (uint32_t)val);
    return *address;
}
template <>
inline __device__ double nd4j_atomicAdd<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <>
inline __device__ Nd4jLong nd4j_atomicAdd<Nd4jLong>(Nd4jLong* address, Nd4jLong val)  {
	unsigned long long int* address_as_ull = (unsigned long long int *) address;

	//return (Nd4jLong) atomicAdd(address_as_ull, (unsigned long long int) val);
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, val + assumed);
	} while (assumed != old);
	return old;
}

template <>
inline __device__ long nd4j_atomicAdd<long>(long* address, long val)  {
	unsigned long long* address_as_ull = (unsigned long long int *) address;

//	return atomicAdd(address, val);
	unsigned long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, val + assumed);
	} while (assumed != old);
	return old;
}

template <>
inline __device__ unsigned long nd4j_atomicAdd<unsigned long>(unsigned long* address, unsigned long val)  {
	unsigned long long* address_as_ull = (unsigned long long int *) address;

//	return atomicAdd(address, val);
	unsigned long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, val + assumed);
	} while (assumed != old);
	return old;
}
template <>
inline __device__ unsigned long long nd4j_atomicAdd<unsigned long long>(unsigned long long* address, unsigned long long val)  {
	//unsigned long* address_as_ull = (unsigned long int *) address;

	//return (Nd4jLong) atomicAdd(address_as_ull, (unsigned long long int) val);
	unsigned long int old = *address, assumed;
	do {
		assumed = old;
		old = atomicCAS(address, assumed, val + assumed);
	} while (assumed != old);
	return old;
}

template <>
inline __device__ float16 nd4j_atomicAdd<float16>(float16* address, float16 val)  {
	int* address_as_ull = (int*) address;

	long addr = (long) address;
	bool misaligned = addr & 0x3;

	if (misaligned)
		address_as_ull = (int *) (addr - 2);

	PAIR old, assumed, fresh;

	old.W = *address_as_ull;
	do {

		if (!misaligned) {
			float16 res = ((float16) old.B.H) + val;
			fresh.B.H = res.data;
			fresh.B.L = old.B.L;
		} else {
			float16 res = ((float16) old.B.L) + val;
			fresh.B.L = res.data;
			fresh.B.H = old.B.H;
		}

		assumed.W = old.W;
		old.W = atomicCAS(address_as_ull, assumed.W, fresh.W);
	} while (assumed.W != old.W);

	if (!misaligned) return old.B.H;
	else return old.B.L;
}

template <>
inline __device__ bfloat16 nd4j_atomicAdd<bfloat16>(bfloat16* address, bfloat16 val)  {
	int* address_as_ull = (int*) address;

	long addr = (long)(address);
	bool misaligned = addr & 0x3;

	if (misaligned)
		address_as_ull = (int *) (addr - 2);

	BPAIR old, assumed, fresh;

	old.W = *address_as_ull;
	do {

		if (!misaligned) {
			bfloat16 res = old.B.H + val;
			fresh.B.H = res;
			fresh.B.L = old.B.L;
		} else {
			bfloat16 res = old.B.L + val;
			fresh.B.L = res;
			fresh.B.H = old.B.H;
		}

		assumed.W = old.W;
		old.W = atomicCAS(address_as_ull, assumed.W, fresh.W);
	} while (assumed.W != old.W);

	if (!misaligned) return old.B.H;
	else return old.B.L;
}

template <>
inline __device__ int16_t nd4j_atomicAdd<int16_t>(int16_t* address, int16_t val)  {
    return nd4j_atomicAdd((bfloat16*)address, (bfloat16)val);
}

template <>
inline __device__ uint16_t nd4j_atomicAdd<uint16_t>(uint16_t* address, uint16_t val)  {
    return nd4j_atomicAdd((bfloat16*)address, (bfloat16)val);
}
template <>
inline __device__ int8_t nd4j_atomicAdd<int8_t>(int8_t* address, int8_t val)  {
    int res = *address;
    atomicAdd(&res, (int)val);
    *address = res;
    return *address;
}

template <>
inline __device__ uint8_t nd4j_atomicAdd<uint8_t>(uint8_t* address, uint8_t val)  {
    int res = *address;
    atomicAdd(&res, (int)val);
    *address = res;
    return *address;
}

template <>
inline __device__ double nd4j_atomicSub<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val -
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <>
inline __device__ double nd4j_atomicMul<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val *
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <>
inline __device__ double nd4j_atomicDiv<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val /
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <>
inline __device__ float nd4j_atomicAdd<float>(float* address, float val)  {
	return atomicAdd(address,val);
}
//template <>
//inline __device__ int nd4j_atomicAdd<int>(int* address, int val)  {
//	return atomicAdd(address, val);
//}
template <>
inline __device__ int32_t nd4j_atomicAdd<int32_t>(int32_t* address, int32_t val)  {
	return (int32_t)atomicAdd((int*)address, (int)val);
}


template <>
inline __device__ float nd4j_atomicSub<float>(float* address, float val) {
	int* address_as_ull = (int*) address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val -
				__float_as_int(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

template <>
inline __device__ float nd4j_atomicMul<float>(float* address, float val) {
	int* address_as_ull =
			( int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val *
				__float_as_int(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

template <>
inline __device__ bfloat16 nd4j_atomicMul<bfloat16>(bfloat16* address, bfloat16 val) {
	int* address_as_ull = (int*) address;

	long addr = (long)(address);
	bool misaligned = addr & 0x3;

	if (misaligned)
		address_as_ull = (int *) (addr - 2);

	BPAIR old, assumed, fresh;

	old.W = *address_as_ull;
	do {

		if (!misaligned) {
			bfloat16 res = old.B.H * val;
			fresh.B.H = res;
			fresh.B.L = old.B.L;
		} else {
			bfloat16 res = old.B.L * val;
			fresh.B.L = res;
			fresh.B.H = old.B.H;
		}

		assumed.W = old.W;
		old.W = atomicCAS(address_as_ull, assumed.W, fresh.W);
	} while (assumed.W != old.W);

	if (!misaligned) return old.B.H;
	else return old.B.L;
}

template <>
inline __device__ float16 nd4j_atomicMul<float16>(float16* address, float16 val) {
	int* address_as_ull = (int*) address;

	long addr = (long)(address);
	bool misaligned = addr & 0x3;

	if (misaligned)
		address_as_ull = (int *) (addr - 2);

	BPAIR old, assumed, fresh;

	old.W = *address_as_ull;
	do {

		if (!misaligned) {
			bfloat16 res = old.B.H * val;
			fresh.B.H = res;
			fresh.B.L = old.B.L;
		} else {
			bfloat16 res = old.B.L * val;
			fresh.B.L = res;
			fresh.B.H = old.B.H;
		}

		assumed.W = old.W;
		old.W = atomicCAS(address_as_ull, assumed.W, fresh.W);
	} while (assumed.W != old.W);

	if (!misaligned) return old.B.H;
	else return old.B.L;
}

template <>
inline __device__ float nd4j_atomicDiv<float>(float* address, float val) {
	int* address_as_ull =
			(int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val *
				__float_as_int(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}
}
#endif
	}

}

#endif /* TEMPLATEMATH_H_ */