/*
 * reduce_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#include <reduce.h>

/**
 *
 * @param extraParams
 * @param sPartials
 * @param sMemSize
 */
template<typename T>
__device__ void functions::reduce::initializeShared(T *extraParams, T **sPartials, int sMemSize) {
	int sPartialsLength = sMemSize / sizeof(T);
	T *sPartialsDeref = (T *) *sPartials;
	for (int i = 0; i < sPartialsLength; i++) {
		sPartialsDeref[i] = extraParams[0];
	}
}










template <>
functions::ops::OpFactory<double> * functions::reduce::ops::getOpFactory<double>() {
   return new functions::reduce::ops::ReduceOpFactoryDouble();
}







