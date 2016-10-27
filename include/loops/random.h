//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_H
#define LIBND4J_RANDOM_H

#define RANDOM_OPS \
        (0, randomOps::BoundedDistribution) ,\
        (1, randomOps::DropOut) ,\
        (2, randomOps::DropOutInverted) ,\
        (3, randomOps::ProbablisticMerge)

#include <shape.h>
#include <helpers/helper_random.h>
#include <ops/random_ops.h>



namespace functions {
    namespace random {

        template<typename T>
        class RandomFunction {
        public:

            template<typename OpClass>
            static inline void execTransform(T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {

            }


            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {

            }

            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
                int length = shape::length(zShapeBuffer);
                int ews = shape::elementWiseStride(zShapeBuffer);

                if (ews >= 1) {
                    if (ews == 1) {
#pragma omp parallel for schedule(guided)
                        for (int x = 0; x < length; x++) {
                            z[x] = OpClass::op(z[x], x, nullptr, extraArguments);
                        }

                    } else {
#pragma omp parallel for schedule(guided)
                        for (int x = 0; x < length; x++) {
                            z[x * ews] = OpClass::op(z[x * ews], x, nullptr, extraArguments);
                        }
                    }
                } else {
                    // ind2sub branch

                    for (int x = 0; x < length; x++) {

                    }
                }
            }


            static inline void execTransform(int opNum, Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
                DISPATCH_BY_OPNUM(execTransform, PARAMS(state, z, zShapeBuffer, extraArguments), RANDOM_OPS)
            }
        };
    }
}

#endif //LIBND4J_RANDOM_H
