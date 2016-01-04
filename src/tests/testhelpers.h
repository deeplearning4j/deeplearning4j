/*
 * testhelpers.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TESTHELPERS_H_
#define TESTHELPERS_H_

#include <templatemath.h>
int arrsEquals(int rank,int *comp1,int *comp2);

template <typename T>
int arrsEquals(int rank, T *comp1,T *comp2) {
	int ret = 1;
	for(int i = 0; i < rank; i++) {
		T eps = nd4j::math::nd4j_abs(comp1[i] - comp2[i]);
		ret = ret && (eps < 1e-3);
	}

	return ret;
}

#endif /* TESTHELPERS_H_ */
