/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_
namespace functions {
namespace reduce3 {

template<typename T>
class Reduce3 {

public:
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//an op for the kernel
	virtual __device__ __host__

	T op(T d1, T d2, T *extraParams);

	//calculate an update of the reduce operation
	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual __device__ __host__

	T update(T old, T opOutput, T *extraParams);


	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual __device__ __host__ T
	merge(T
			old,
			T opOutput, T
			*extraParams);

	/**
	 *
	 * @param n
	 * @param sPartials
	 * @param dx
	 * @param xOffset
	 * @param incx
	 * @param dy
	 * @param yOffset
	 * @param incy
	 * @param extraParams
	 * @return
	 */
	virtual __device__ T

	doBlock(
			int n,
			T *sPartials,
			T *dx,
			int xOffset,
			int incx,
			T *dy,
			int yOffset,
			int incy,
			T *extraParams);


	/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(T **sPartialsRef, int tid, T *extraParams);

	/**
	 *
	 * @param n
	 * @param dx
	 * @param xShapeInfo
	 * @param dy
	 * @param yShapeInfo
	 * @param extraParams
	 * @param result
	 * @param resultShapeInfo
	 * @param gpuInformation
	 * @param dimension
	 * @param dimensionLength
	 * @param postProcessOrNot
	 */
	virtual __device__ void transform(
			int n, T *dx, int *xShapeInfo,
			T *dy,
			int *yShapeInfo, T *extraParams, T *result,
			int *resultShapeInfo, int *gpuInformation,
			int *dimension,
			int dimensionLength, int postProcessOrNot);

	virtual ~Reduce3();


};


}
}


#endif /* REDUCE3_H_ */
