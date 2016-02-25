/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
namespace functions {
    namespace broadcast {

        template<typename T>
        class Broadcast {
        public:

            /**
             *
             * @param d1
             * @param d2
             * @return
             */
            virtual __device__ T
            op(T
            d1,
            T d2
            );
            /**
             *
             * @param d1
             * @return
             */
            virtual __device__ T
            op(T
            d1);

            /**
             *
             * @param x
             * @param xShapeInfo
             * @param y
             * @param yShapeInfo
             * @param result
             * @param resultShapeInfo
             * @param dimension
             * @param dimensionLength
             * @param gpuInformation
             */
            virtual __device__ void transform(
                    T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result, int *resultShapeInfo,
                    int *dimension,
                    int dimensionLength,
                    int *gpuInformation);

            virtual ~Broadcast();

        };
    }
}


#endif /* BROADCASTING_H_ */
