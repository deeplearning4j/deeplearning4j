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
		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		 T nd4j_abs(T value);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_max(T val1, T val2);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_min(T val1, T val2);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_ceil(T val1);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_cos(T val);



		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_exp(T val);


		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_floor(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_log(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_pow(T val, T val2);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_round(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_sigmoid(T val) {
			return 1.0 / (1.0 + nd4j_exp<T>(-val));
		}

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_sin(T val);


		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T softplus(T val) {
			return nd4j_log<T>(1 + nd4j_exp<T>(val));
		}

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_sqrt(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_tanh(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_acos(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
                __always_inline
#endif
		T nd4j_asin(T val);

		template <typename T>
#ifdef __CUDACC__
		__host__ __device__
#elseif __GNUC__
		__always_inline
#endif
		T nd4j_atan(T val);


	}
}




#endif /* TEMPLATEMATH_H_ */
