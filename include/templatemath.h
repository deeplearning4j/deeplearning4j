/*
 * templatemath.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TEMPLATEMATH_H_
#define TEMPLATEMATH_H_

#include <math.h>

namespace nd4j {
	namespace math {
		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_abs(T value);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_max(T val1, T val2);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_min(T val1, T val2);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_ceil(T val1);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_cos(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_exp(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_floor(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_log(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_pow(T val, T val2);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_round(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_sigmoid(T val) {
			return 1.0 / (1.0 + nd4j_exp<T>(-val));
		}

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_elu(T val) {
			return val >= 0.0 ? val : (nd4j_exp<T>(val) - 1.0);
		}


        template<typename T>
#ifdef __CUDACC__
        __host__ __device__

#endif
        inline T nd4j_leakyrelu(T val,T alpha) {
            return val < 0 ?  alpha * val : val;
        }


        template<typename T>
#ifdef __CUDACC__
        __host__ __device__

#endif
        inline T nd4j_eluderivative(T val) {
            return val >= 0.0 ? 1.0 : nd4j_exp(val);
        }
		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_sin(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T softplus(T val) {
			return nd4j_log<T>(1.0 + nd4j_exp<T>(val));
		}
		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_softsign(T val) {
			return val / (1.0 + nd4j::math::nd4j_abs<T>(val));
		}

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_sqrt(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_tanh(T val);
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__

#endif
        inline T nd4j_tanhderivative(T val) {
            T tanh = nd4j_tanh(val);
            return 1.0 - tanh * tanh;
        }
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__

#endif
        inline T nd4j_sigmoidderivative(T val) {
            T sigmoid = nd4j_sigmoid(val);
            T out = sigmoid * (1.0 - sigmoid);
            return out;
        }

        template<typename T>
#ifdef __CUDACC__
        __host__ __device__

#endif
        inline T nd4j_softsignderivative(T val) {
            T y = 1 + nd4j_abs(val);
            return 1.0 / (y * y);
        }
		template<typename T>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline T nd4j_acos(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_asin(T val);

		template<typename T>
#ifdef __CUDACC__
		__host__ __device__

#endif
		inline T nd4j_atan(T val);

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_abs<float>(float value) {
			return fabsf(value);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_abs<double>(double value) {
			return value < 0 ? -value : value;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_abs<int>(int value) {
			return value < 0 ? -value : value;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_max<float>(float val1, float val2) {
			return val1 > val2 ? val1 : val2;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_max<double>(double val1, double val2) {
			return val1 > val2 ? val1 : val2;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_max<int>(int val1, int val2) {
			return val1 > val2 ? val1 : val2;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_min<float>(float val1, float val2) {
			return val1 < val2 ? val1 : val2;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_min<double>(double val1, double val2) {
			return val1 < val2 ? val1 : val2;
		}
		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_min<int>(int val1, int val2) {
			return val1 < val2 ? val1 : val2;
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif

		inline float nd4j_ceil<float>(float val1) {
			return ceilf(val1);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif

		inline double nd4j_ceil<double>(double val) {
			return ceil(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_ceil<int>(int val) {
			return ceil((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_cos<float>(float val) {
			return cosf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_cos<double>(double val) {
			return cos(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_cos<int>(int val) {
			return cosf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_exp<float>(float val) {
			return expf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_exp<double>(double val) {
			return exp(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_exp<int>(int val) {
			return expf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_floor<float>(float val) {
			return floorf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_floor<double>(double val) {
			return floor(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_floor<int>(int val) {
			return floorf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_log<float>(float val) {
			return logf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_log<double>(double val) {
			return log(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_log<int>(int val) {
			return logf((int) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_pow<float>(float val, float val2) {
			return powf(val, val2);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_pow<double>(double val, double val2) {
			return pow(val, val2);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_pow<int>(int val, int val2) {
			return powf((float) val, (float) val2);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_round<float>(float val) {
			return roundf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_round<double>(double val) {
			return round(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_round<int>(int val) {
			return round((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_sin<float>(float val) {
			return sinf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_sin<double>(double val) {
			return sin(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_sin<int>(int val) {
			return sin((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_sqrt<float>(float val) {
			return sqrtf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_sqrt<double>(double val) {
			return sqrt(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_sqrt<int>(int val) {
			return sqrtf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_tanh<float>(float val) {
			return tanhf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_tanh<double>(double val) {
			return tanh(val);
		}
		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_tanh<int>(int val) {
			return tanhf((float) val);
		}
		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_acos<float>(float val) {
			return acosf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_acos<double>(double val) {
			return acos(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_acos<int>(int val) {
			return acosf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_asin<float>(float val) {
			return asinf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_asin<double>(double val) {
			return asin(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_asin<int>(int val) {
			return asinf((float) val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline float nd4j_atan<float>(float val) {
			return atanf(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline double nd4j_atan<double>(double val) {
			return atan(val);
		}

		template<>
#ifdef __CUDACC__
		__host__ __device__
#endif
		inline int nd4j_atan<int>(int val) {
			return atanf((float) val);
		}

#ifdef __CUDACC__
		namespace atomics {
template <typename T>
__device__ T nd4j_atomicAdd(T* address, T val);
template <typename T>
__device__ T nd4j_atomicSub(T* address, T val);
template <typename T>
__device__ T nd4j_atomicMul(T* address, T val);
template <typename T>
__device__ T nd4j_atomicDiv(T* address, T val);

template <>
__device__ double nd4j_atomicAdd<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_long longlong long(val +
				__long longlong long_as_double(assumed)));
	} while (assumed != old);
	return __long longlong long_as_double(old);
}

template <>
__device__ double nd4j_atomicSub<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_long longlong long(val -
				__long longlong long_as_double(assumed)));
	} while (assumed != old);
	return __long longlong long_as_double(old);
}

template <>
__device__ double nd4j_atomicMul<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_long longlong long(val *
				__long longlong long_as_double(assumed)));
	} while (assumed != old);
	return __long longlong long_as_double(old);
}

template <>
__device__ double nd4j_atomicDiv<double>(double* address, double val)  {
	unsigned long long int* address_as_ull =
			(unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_long longlong long(val /
				__long longlong long_as_double(assumed)));
	} while (assumed != old);
	return __long longlong long_as_double(old);
}

template <>
__device__ float nd4j_atomicAdd<float>(float* address, float val)  {
	return atomicAdd(address,val);
}

template <>
__device__ float nd4j_atomicSub<float>(float* address, float val) {
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
__device__ float nd4j_atomicMul<float>(float* address, float val) {
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
__device__ float nd4j_atomicDiv<float>(float* address, float val) {
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
