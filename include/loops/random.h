//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_H
#define LIBND4J_RANDOM_H



#include <helpers/shape.h>
#include <helpers/helper_random.h>
#include <ops/random_ops.h>
#include <ops/special_random_ops.h>

#include <loops/legacy_ops.h>


namespace functions {
    namespace random {

        template<typename T>
        class RandomFunction {
        public:

#ifdef __CUDACC__
            template<typename OpClass>
            static _CUDA_D void execTransformCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);

            template<typename OpClass>
            static _CUDA_D void execTransformCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);

            template<typename OpClass>
            static _CUDA_D void execTransformCuda(Nd4jPointer state, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);


            static _CUDA_H void executeCudaSingle(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
            static _CUDA_H void executeCudaDouble(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, T *x, Nd4jLong *xShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
            static _CUDA_H void executeCudaTriple(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
#endif

            template<typename OpClass>
            static void execTransform(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);

            template<typename OpClass>
            static void execTransform(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);

            template<typename OpClass>
            static void execTransform(Nd4jPointer state, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);

            static void execTransform(int opNum, Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
            static void execTransform(int opNum, Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
            static void execTransform(int opNum, Nd4jPointer state, T *z, Nd4jLong *zShapeBuffer, T *extraArguments);
        };
    }
}


#endif //LIBND4J_RANDOM_H
