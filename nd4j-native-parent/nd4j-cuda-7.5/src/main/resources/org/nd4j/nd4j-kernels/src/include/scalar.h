/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_
namespace functions {
    namespace scalar {
        template<typename T>
        class ScalarTransform {

        public:
            /**
             *
             * @param d1
             * @param d2
             * @param params
             * @return
             */
            virtual __device__ __host__ T
            op(T
            d1,
            T d2, T
            *params);

            /**
             *
             * @param n
             * @param idx
             * @param dx
             * @param dy
             * @param incy
             * @param params
             * @param result
             * @param blockSize
             */
            virtual __device__ void transform(int n, int idx, T dx, T *dy, int incy, T *params, T *result,
                                              int blockSize);

            virtual ~ScalarTransform();
        };


    }
}


#endif /* SCALAR_H_ */
