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

#include <math.h>
#include <cmath>
#include <dll.h>
#include <pointercast.h>


#define HALF_MAX_VALUE 65504.
#define FLOAT_MAX_VALUE 3.4028235E38
#define DOUBLE_MAX_VALUE 1.7976931348623157E308
#define FLOAT_MIN_NORMAL 1.17549435e-38

#ifndef M_E
#define M_E 2.718281828459
#endif

#ifdef __CUDACC__
#include <types/float16.h>
#define math_def __host__ __device__
#ifdef CUDA_9
struct HALFS{
			half H;
			half L;

            __host__ __device__
			HALFS() {};

			__host__ __device__
			~HALFS() {};
		};
union PAIR {
		HALFS B;
		int W;

        __host__ __device__
		PAIR() {};

		__host__ __device__
		~PAIR(){}

};
#else
typedef union {
        struct {
            half H;
            half L;
        } B;
        int W;
} PAIR;
#endif // cuda_9

#else
#define math_def
#include <types/float16.h>
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

		template<typename T>
		math_def inline T nd4j_re(T val1, T val2);

		template<typename T>
        math_def inline T nd4j_rint(T val1);

		template<typename T>
		math_def inline T nd4j_copysign(T val1, T val2);

//#ifndef __CUDACC__
        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_dot(X *x, Y *y, int length);
//#endif

		template<typename T>
        math_def inline T nd4j_ceil(T val1);

		template<typename T>
        math_def inline bool nd4j_isnan(T val1);

		template<typename T>
        math_def inline bool nd4j_isinf(T val1);

		template<typename T>
        math_def inline bool nd4j_isfin(T val1);

		template<typename T>
        math_def inline T nd4j_cos(T val);

        template<typename T>
        math_def inline T nd4j_cosh(T val);

		template<typename X, typename Z>
        math_def inline Z nd4j_exp(X val);

		template<typename T>
        math_def inline T nd4j_floor(T val);

		template<typename X, typename Z>
        math_def inline Z nd4j_log(X val);

		template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_pow(X val, Y val2);

		template<typename T>
        math_def inline T nd4j_round(T val);

        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_remainder(X num, Y denom);

        template<typename X, typename Y, typename Z>
        math_def inline Z nd4j_fmod(X num, Y denom);

		template<typename T>
        math_def inline T nd4j_erf(T num);

		template<typename T>
        math_def inline T nd4j_erfc(T num);

		template<typename T>
        math_def inline T nd4j_sigmoid(T val) {
			return (T) 1.0 / ((T) 1.0 + nd4j_exp<T, T>(-val));
		}

		template<typename T>
        math_def inline T nd4j_elu(T val) {
			if (val >= (T) 0.0) return val;
			else return nd4j_exp<T, T>(val) - (T) 1.0;
			//return val >= 0.0 ? val : (nd4j_exp<T>(val) - 1.0);
		}


		template<typename T>
        math_def inline T nd4j_leakyrelu(T val,T alpha) {
			if (val < (T) 0.0f) return alpha * val;
			else return val;
			//return val < 0 ?  alpha * val : val;
		}


		template<typename T>
        math_def inline T nd4j_eluderivative(T val) {
			if (val >= (T) 0.0f) return (T) 1.0f;
			else return nd4j_exp<T, T>(val);
			//return val >= 0.0 ? 1.0 : nd4j_exp(val);
		}
		template<typename T>
        math_def inline T nd4j_sin(T val);

		template<typename T>
		math_def inline T nd4j_sinh(T val);
		template<typename T>
        math_def inline T softplus(T val) {
			return nd4j_log<T, T>((T) 1.0f + nd4j_exp<T, T>(val));
		}
		template<typename T>
        math_def inline T nd4j_softsign(T val) {
			return val / ((T) 1.0f + nd4j::math::nd4j_abs<T>(val));
		}

		template<typename X, typename Z>
        math_def inline Z nd4j_sqrt(X val);

		template<typename T>
        math_def inline T nd4j_tanh(T val);
        template<typename T>
        math_def inline T nd4j_tan(T val);

		template<typename T>
		math_def inline T nd4j_atan2(T val1, T val2);

		template<>
		math_def inline float16 nd4j_atan2<float16>(float16 value1, float16 value2) {
			return (float16) atan2f((float) value1, (float) value2);
		}

		template<>
		math_def inline float nd4j_atan2<float>(float value1, float value2) {
			return atan2f(value1, value2);
		}

		template<>
		math_def inline double nd4j_atan2<double>(double value1, double value2) {
			return atan2(value1, value2);
		}

		template<>
		math_def inline int nd4j_atan2<int>(int value1, int value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline uint8_t nd4j_atan2<uint8_t>(uint8_t value1, uint8_t value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline bool nd4j_atan2<bool>(bool value1, bool value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline uint16_t nd4j_atan2<uint16_t>(uint16_t value1, uint16_t value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline uint32_t nd4j_atan2<uint32_t>(uint32_t value1, uint32_t value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline int8_t nd4j_atan2<int8_t>(int8_t value1, int8_t value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline int16_t nd4j_atan2<int16_t>(int16_t value1, int16_t value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline Nd4jLong nd4j_atan2<Nd4jLong>(Nd4jLong value1, Nd4jLong value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

		template<>
		math_def inline Nd4jULong nd4j_atan2<Nd4jULong>(Nd4jULong value1, Nd4jULong value2) {
			return atan2f(static_cast<float>(value1), static_cast<float>(value2));
		}

        template<typename T>
        math_def inline T nd4j_tan(T val) {
            return nd4j_log<T, T>((val + 1 / (1 - val)) * 0.5);
        }

        template<typename T>
        math_def inline T nd4j_tanhderivative(T val) {
			T tanh = nd4j_tanh(val);
			return (T) 1.0f - tanh * tanh;
		}
		template<typename T>
        math_def inline T nd4j_sigmoidderivative(T val) {
			T sigmoid = nd4j_sigmoid(val);
			T out = sigmoid * ((T) 1.0f - sigmoid);
			return out;
		}

		template<typename T>
        math_def inline T nd4j_softsignderivative(T val) {
			T y = (T) 1.0f + nd4j_abs(val);
			return (T) 1.0f / (y * y);
		}

        template<typename T>
        math_def inline T nd4j_sgn(T val) {
            return val < (T) 0.0f ? (T) -1.0f : val > (T) 0.0f ? (T) 1.0f : (T) 0.0f;
        }

        template<typename T>
        math_def inline T nd4j_sign(T val) {
            return nd4j_sgn<T>(val);
        }

        template<typename T>
        math_def inline T nd4j_signum(T val) {
            return nd4j_sgn<T>(val);
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

//#pragma omp simd reduction(+:dot)
			for(int e = 0; e < length; e++) {
				dot += x[e] * y[e];
			}

			return dot;
		}
//#endif

		template<typename T>
        math_def inline T nd4j_acos(T val);

		template<typename T>
		math_def inline T nd4j_acosh(T val);

		template<typename T>
        math_def inline T nd4j_asin(T val);

		template<typename T>
		math_def inline T nd4j_asinh(T val);

        template<typename T>
        math_def inline T nd4j_asinh(T val) {
            //Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x)
            return nd4j_log<T, T>(nd4j_sqrt<T, T>(nd4j_pow<T,T,T>(val, (T) 2) + (T) 1) + val);
        }

		template<typename T>
        math_def inline T nd4j_atan(T val);

        template<typename T>
        math_def inline T nd4j_atanh(T val);


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
        math_def inline float16 nd4j_rint<float16>(float16 value) {
			return (float16) rintf((float) value);
		}


		template<>
        math_def inline float nd4j_rint<float>(float value) {
			return rintf(value);
		}

		template<>
        math_def inline double nd4j_rint<double>(double value) {
			return rint(value);
		}

		template<>
        math_def inline int nd4j_rint<int>(int value) {
			return value;
		}

		template<>
		math_def inline Nd4jLong nd4j_rint<Nd4jLong>(Nd4jLong value) {
			return value;
		}

		template<>
        math_def inline bool nd4j_isnan<float16>(float16 value) {
			return *(value.data.getXP()) == 0x7fffU;
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


		template<typename T>
		math_def inline T nd4j_max(T val1, T val2) {
			return val1 > val2 ? val1 : val2;
		}

		template<typename T>
		math_def inline T nd4j_min(T val1, T val2) {
			return val1 < val2 ? val1 : val2;
		}


		template<>
        math_def inline float16 nd4j_ceil<float16>(float16 val) {
#ifdef NATIVE_HALFS
            return hceil(val.data);
#else
			return ceilf((float) val);
#endif
		}

		template<>
        math_def inline float nd4j_ceil<float>(float val1) {
			return ceilf(val1);
		}

		template<>
        math_def inline double nd4j_ceil<double>(double val) {
			return ceil(val);
		}

		template<>
        math_def inline int nd4j_ceil<int>(int val) {
			return ceil((float) val);
		}

		template<>
        math_def inline float16 nd4j_cos<float16>(float16 val) {
#ifdef NATIVE_HALFS
			return hcos(val.data);
#else
			return cosf((float) val);
#endif
		}

		template<>
        math_def inline float nd4j_cos<float>(float val) {
			return cosf(val);
		}

		template<>
        math_def inline double nd4j_cos<double>(double val) {
			return cos(val);
		}

		template<>
        math_def inline int nd4j_cos<int>(int val) {
			return cosf((float) val);
		}


        template<>
        math_def inline float16 nd4j_cosh<float16>(float16 val) {
            return coshf((float) val);
        }

        template<>
        math_def inline float nd4j_cosh<float>(float val) {
            return coshf(val);
        }

        template<>
        math_def inline double nd4j_cosh<double>(double val) {
            return cosh(val);
        }

        template<>
        math_def inline int nd4j_cosh<int>(int val) {
            return coshf((float) val);
        }


		template <typename X, typename Z>
        math_def inline Z nd4j_exp(X val) {
            if (std::is_same<X, float16>::value) {
#ifdef NATIVE_HALFS
                return hexp(val.data);
#else
                return static_cast<Z>(expf((float) val));
#endif
            } else if (std::is_same<X, double>::value) {
                return static_cast<Z>(exp(val));
            } else if (std::is_same<X, float>::value) {
                return static_cast<Z>(sqrtf(val));
            } else {
                return static_cast<Z>(sqrtf((float) val));
            }
        }

		template<>
        math_def inline float16 nd4j_floor<float16>(float16 val) {
#ifdef NATIVE_HALFS
            return hfloor(val.data);
#else
			return (float16) floorf((float) val);
#endif
		}


		template<>
        math_def inline float nd4j_floor<float>(float val) {
			return floorf(val);
		}

		template<>
        math_def inline double nd4j_floor<double>(double val) {
			return floor(val);
		}

		template<>
        math_def inline int nd4j_floor<int>(int val) {
			return val;
		}

		template<>
		math_def inline int8_t nd4j_floor<int8_t>(int8_t val) {
			return val;
		}

		template<>
		math_def inline bool nd4j_floor<bool>(bool val) {
			return val;
		}

		template<>
		math_def inline uint8_t nd4j_floor<uint8_t>(uint8_t val) {
			return val;
		}

		template<>
		math_def inline int16_t nd4j_floor<int16_t>(int16_t val) {
			return val;
		}

		template<>
		math_def inline uint16_t nd4j_floor<uint16_t>(uint16_t val) {
			return val;
		}

		template<>
		math_def inline uint32_t nd4j_floor<uint32_t>(uint32_t val) {
			return val;
		}

		template<>
		math_def inline Nd4jULong nd4j_floor<Nd4jULong>(Nd4jULong val) {
			return val;
		}

		template<>
		math_def inline Nd4jLong nd4j_floor<Nd4jLong>(Nd4jLong val) {
			return val;
		}

		template<typename X, typename Z>
        math_def inline Z nd4j_log(X val) {
        	if (std::is_same<X, float16>::value) {
#ifdef NATIVE_HALFS
				return static_cast<Z>(hlog(val.data));
#else
				return static_cast<Z>(logf((float) val));
#endif
			} else if (std::is_same<X, double>::value) {
				return static_cast<Z>(log(val));
        	} else if (std::is_same<X, float>::value) {
				return static_cast<Z>(logf(val));
        	} else {
				return static_cast<Z>(logf((float) val));
        	}
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
        	if (std::is_same<X, double>::value || std::is_same<Y, double>::value) {
				return static_cast<Z>(pow(static_cast<double>(val), static_cast<double>(val2)));
			} else {
				return static_cast<Z>(powf(static_cast<float>(val), static_cast<float>(val2)));
        	}
		}



		template<typename T>
		math_def inline T nd4j_re(T val1, T val2) {
			if (val1 == (T) 0.0f && val2 == (T) 0.0f)
				return (T) 0.0f;

			return nd4j_abs<T>(val1 - val2) / (nd4j_abs<T>(val1) + nd4j_abs<T>(val2));
        }

		template<>
        math_def inline float16 nd4j_round<float16>(float16 val) {
			return (float16) roundf((float) val);
		}


		template<>
        math_def inline float nd4j_round<float>(float val) {
			return roundf(val);
		}

		template <typename X, typename Y, typename Z>
		math_def inline Z nd4j_remainder(X val, Y val2) {
			if (std::is_same<X, double>::value || std::is_same<Y, double>::value) {
				return static_cast<Z>(remainder(static_cast<double>(val), static_cast<double>(val2)));
			} else {
				return static_cast<Z>(remainderf(static_cast<float>(val), static_cast<float>(val2)));
			}
		}

		template <typename X, typename Y, typename Z>
		math_def inline Z nd4j_fmod(X val, Y val2) {
			if (std::is_same<X, double>::value || std::is_same<Y, double>::value) {
				return static_cast<Z>(fmod(static_cast<double>(val), static_cast<double>(val2)));
			} else {
				return static_cast<Z>(fmodf(static_cast<float>(val), static_cast<float>(val2)));
			}
		}

		template<>
        math_def inline float nd4j_erf<float>(float num) {
            return erff(num);
        }

        template<>
        math_def inline double nd4j_erf<double>(double num) {
            return erf(num);
        }

        template<>
        math_def inline float16 nd4j_erf<float16>(float16 num) {
            return (float16) erff((float) num);
        }

		template<>
        math_def inline float nd4j_erfc<float>(float num) {
            return erfcf(num);
        }

        template<>
        math_def inline double nd4j_erfc<double>(double num) {
            return erfc(num);
        }

        template<>
        math_def inline float16 nd4j_erfc<float16>(float16 num) {
            return (float16) erfcf((float) num);
        }



		template<>
        math_def inline double nd4j_round<double>(double val) {
			return round(val);
		}

		template<>
        math_def inline int nd4j_round<int>(int val) {
			return round((float) val);
		}

		template<>
        math_def inline float16 nd4j_sin<float16>(float16 val) {
#ifdef NATIVE_HALFS
            return hsin(val.data);
#else
			return (float16) sinf((float) val);
#endif
		}


		template<>
        math_def inline float nd4j_sin<float>(float val) {
			return sinf(val);
		}

		template<>
        math_def inline double nd4j_sin<double>(double val) {
			return sin(val);
		}

		template<>
        math_def inline int nd4j_sin<int>(int val) {
			return sin((float) val);
		}



		template<>
		math_def inline float16 nd4j_sinh<float16>(float16 val) {
#ifdef NATIVE_HALFS
			return hsin(val.data);
#else
			return (float16) sinh((float) val);
#endif
		}


		template<>
		math_def inline float nd4j_sinh<float>(float val) {
			return sinhf(val);
		}

		template<>
		math_def inline double nd4j_sinh<double>(double val) {
			return sinh(val);
		}

		template<>
		math_def inline int nd4j_sinh<int>(int val) {
			return sinhf((float) val);
		}


		template <typename X, typename Z>
        math_def inline Z nd4j_sqrt(X val) {
            if (std::is_same<X, float16>::value) {
#ifdef NATIVE_HALFS
                return static_cast<Z>(hsqrt(val.data));
#else
                return static_cast<Z>(sqrtf((float) val));
#endif
            } else if (std::is_same<X, double>::value) {
                return static_cast<Z>(sqrt(val));
            } else if (std::is_same<X, float>::value) {
                return static_cast<Z>(sqrtf(val));
            } else {
                return static_cast<Z>(sqrt((float) val));
            }
        }

		template<>
        math_def inline float16 nd4j_tanh<float16>(float16 val) {
			return (float16) tanhf((float) val);
		}


		template<>
        math_def inline float nd4j_tanh<float>(float val) {
			return tanhf(val);
		}

		template<>
        math_def inline double nd4j_tanh<double>(double val) {
			return tanh(val);
		}
		template<>
        math_def inline int nd4j_tanh<int>(int val) {
			return tanhf((float) val);
		}


        template<>
        math_def inline float16 nd4j_tan<float16>(float16 val) {
            return (float16) tanf((float) val);
        }


        template<>
        math_def inline float nd4j_tan<float>(float val) {
            return tanf(val);
        }

        template<>
        math_def inline double nd4j_tan<double>(double val) {
            return tan(val);
        }
        template<>
        math_def inline int nd4j_tan<int>(int val) {
            return tanf((float) val);
        }


        template<>
        math_def inline float16 nd4j_acos<float16>(float16 val) {
			return (float16) acosf((float) val);
		}


		template<>
        math_def inline float nd4j_acos<float>(float val) {
			return acosf(val);
		}

		template<>
        math_def inline double nd4j_acos<double>(double val) {
			return acos(val);
		}

		template<>
        math_def inline int nd4j_acos<int>(int val) {
			return acosf((float) val);
		}


		template<>
		math_def inline float16 nd4j_acosh<float16>(float16 val) {
			return (float16) acoshf((float) val);
		}


		template<>
		math_def inline float nd4j_acosh<float>(float val) {
			return acoshf(val);
		}

		template<>
		math_def inline double nd4j_acosh<double>(double val) {
			return acosh(val);
		}

		template<>
		math_def inline int nd4j_acosh<int>(int val) {
			return acoshf((float) val);
		}


		template<>
        math_def inline float16 nd4j_asin<float16>(float16 val) {
			return (float16) asinf((float) val);
		}


		template<>
        math_def inline float nd4j_asin<float>(float val) {
			return asinf(val);
		}

		template<>
        math_def inline double nd4j_asin<double>(double val) {
			return asin(val);
		}

		template<>
        math_def inline int nd4j_asin<int>(int val) {
			return asinf((float) val);
		}


		template<>
        math_def inline float16 nd4j_atan<float16>(float16 val) {
			return (float16) atanf((float)val);
		}


		template<>
        math_def inline float nd4j_atan<float>(float val) {
			return atanf(val);
		}

		template<>
        math_def inline double nd4j_atan<double>(double val) {
			return atan(val);
		}

		template<>
        math_def inline int nd4j_atan<int>(int val) {
			return atanf((float) val);
		}





        template<>
        math_def inline float16 nd4j_atanh<float16>(float16 val) {
            return (float16) atanhf((float)val);
        }


        template<>
        math_def inline float nd4j_atanh<float>(float val) {
            return atanhf(val);
        }

        template<>
        math_def inline double nd4j_atanh<double>(double val) {
            return atanh(val);
        }

        template<>
        math_def inline int nd4j_atanh<int>(int val) {
            return atanhf((float) val);
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