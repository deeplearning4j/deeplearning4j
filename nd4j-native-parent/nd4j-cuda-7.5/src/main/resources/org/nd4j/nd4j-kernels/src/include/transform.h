/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
namespace nd4j {
    namespace functions {
        namespace transform {
            template<typename T>
            class Transform {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual __device__ __host__

                T op(T d1, T *params);

                /**
                 * Apply a single transform
                 * @param n
                 * @param idx
                 * @param dy
                 * @param incy
                 * @param params
                 * @param result
                 * @param blockSize
                 */
                virtual __device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize);

                virtual ~Transform();

            };

        }
    }
}


#endif /* TRANSFORM_H_ */
