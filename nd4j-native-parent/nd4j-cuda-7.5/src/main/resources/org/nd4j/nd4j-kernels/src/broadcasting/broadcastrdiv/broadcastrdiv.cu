#include <broadcasting.h>
#include <helper_cuda.h>
/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ double op<double>(double d1,double d2) {
	return d2 / d1;
}





/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ float op<float>(float d1,float d2) {
	return d2 / d1;
}







