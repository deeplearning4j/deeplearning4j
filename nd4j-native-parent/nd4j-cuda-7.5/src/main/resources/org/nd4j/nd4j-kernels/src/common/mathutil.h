/*
 * mathtuil.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef MATHTUIL_H_
#define MATHTUIL_H_

/**
 * Returns the prod of the data
 * up to the given length
 */
__device__ __host__ int prod(int *data,int length) {
	int prod = 1;
	for(int i = 0; i < length; i++) {
		prod *= data[i];
	}

	return prod;
}



#endif /* MATHTUIL_H_ */
