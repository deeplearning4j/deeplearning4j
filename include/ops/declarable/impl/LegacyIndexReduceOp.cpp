//
// Created by raver119 on 16.10.2017.
//

#include "ops/declarable/LegacyIndexReduceOp.h"
#include <helpers/ShapeUtils.h>


namespace nd4j {
    namespace ops {


        template <typename T>
        LegacyIndexReduceOp<T>::LegacyIndexReduceOp() : LegacyOp<T>::LegacyOp(1){
            //
        }

        template <typename T>
        LegacyIndexReduceOp<T>::LegacyIndexReduceOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //
        }

        template <typename T>
        ShapeList *LegacyIndexReduceOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;
            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                // in this case we just return scalar
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;
            } else {
                // in this case we're building proper shape for reduction
                auto array = new NDArray<T>(nullptr, inShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                newShape = ShapeUtils<T>::evalReduceShapeInfo('c', *block.getIArguments(), *array);

                delete array;
            }

            return new ShapeList(newShape);
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        Nd4jStatus LegacyIndexReduceOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                // scalar
                T res = NativeOpExcutioner<T>::execIndexReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data());
                z->putScalar(0, res);
            } else {
                // TAD
                std::vector<int> dims(*block.getIArguments());

                if (dims.size() > 1)
                    std::sort(dims.begin(), dims.end());

                shape::TAD tad(x->getShapeInfo(), dims.data(), dims.size());
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                NativeOpExcutioner<T>::execIndexReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo(), dims.data(), (int) dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LegacyIndexReduceOp<float>;
        template class ND4J_EXPORT LegacyIndexReduceOp<double>;
        template class ND4J_EXPORT LegacyIndexReduceOp<float16>;
    }
}