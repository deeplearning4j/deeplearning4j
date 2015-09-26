/*
 * deeplearning4j.h
 *
 *  Created on: Mar 12, 2015
 *      Author: agibsonccc
 */

#ifndef DEEPLEARNING4J_H_
#define DEEPLEARNING4J_H_

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }




#endif /* DEEPLEARNING4J_H_ */
