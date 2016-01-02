/*
 * templatemath.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TEMPLATEMATH_H_
#define TEMPLATEMATH_H_
#include <math.h>

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T abs(T value);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T max(T val1,T val2);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T min(T val1,T val2);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T ceil(T val1);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T cos(T val);



template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T exp(T val);


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T floor(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T log(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T pow(T val,T val2);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T round(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T sigmoid(T val) {
	return 1.0 / (1.0 + exp<T>(-val));
}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T sin(T val);


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T softplus(T val) {
	return log<T>(1 + exp<T>(val));

}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T sqrt(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T tanh(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T acos(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
 T asin(T val);

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
T atan(T val);


template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 float abs<float>(float value) {
	return fabsf(value);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double abs<double>(double value) {
	return abs(value);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float max<float>(float val1,float val2) {
	return fmaxf(val1,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double max<double>(double val1,double val2) {
	return max(val1,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float min<float>(float val1,float val2) {
	return fminf(val1,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double min<double>(double val1,double val2) {
	return min(val1,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float ceil<float>(float val1) {
	return ceilf(val1);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double ceil<double>(double val) {
	return ceil(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float cos<float>(float val) {
	return cosf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double cos<double>(double val) {
	return cos(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float exp<float>(float val) {
	return expf(val);
}
template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double exp<double>(double val) {
	return exp(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float floor<float>(float val) {
	return floor(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double floor<double>(double val) {
	return floor(val);
}


template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float log<float>(float val) {
	return logf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double log<double>(double val) {
	return log(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float pow<float>(float val,float val2) {
	return powf(val,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double pow<double>(double val,double val2) {
	return pow(val,val2);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float round<float>(float val) {
	return roundf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double round<double>(double val) {
	return round(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float sin<float>(float val) {
	return sinf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double sin<double>(double val) {
	return sin(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 float sqrt<float>(float val) {
	return sqrtf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double sqrt<double>(double val) {
	return sqrt(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 float tanh<float>(float val) {
	return tanhf(val);
}


template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double tanh<double>(double val) {
	return tanf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float acos<float>(float val) {
	return acosf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
double acos<double>(double val) {
	return acos(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float asin<float>(float val) {
	return asinf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double asin<double>(double val) {
	return asin(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
float atan<float>(float val) {
	return atanf(val);
}

template <>
#ifdef __CUDACC__
__host__ __device__
#endif
 double atan<double>(double val) {
	return atan(val);
}


#endif /* TEMPLATEMATH_H_ */
