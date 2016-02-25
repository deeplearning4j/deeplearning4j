/*
 * postprocess.h
 *
 *  Created on: Dec 19, 2015
 *      Author: agibsonccc
 */

#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

namespace nd4j {
    namespace functions {
        namespace reduce {


/**
 * Post process n items
 * @param n
 * @param xOffset
 * @param dx
 * @param incx
 * @param extraParams
 * @param result
 */
            template<typename T>
            __device__ void postProcessLoop(int n, int xOffset, T *dx, int incx, T *extraParams, T *result);


        }
    }
}


#endif /* POSTPROCESS_H_ */
