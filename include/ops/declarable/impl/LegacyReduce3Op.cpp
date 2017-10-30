//
// Created by raver119 on 17.10.2017.
//

#include "ops/declarable/LegacyReduce3Op.h"
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        Nd4jStatus LegacyReduce3Op<T>::validateAndExecute(Block<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            if (x->isSameShape(y) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && block.getIArguments()->at(0) == MAX_INT))) {
                // reduce3 to scalar
                T scalar = NativeOpExcutioner<T>::execReduce3Scalar(opNum, x->buffer(), x->shapeInfo(), block.getTArguments()->data(), y->buffer(), y->shapeInfo());
                z->putScalar(0, scalar);
            } else {
                std::vector<int> dims(*block.getIArguments());
                std::sort(dims.begin(), dims.end());

                REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

                NativeOpExcutioner<T>::execReduce3(opNum, x->buffer(), x->shapeInfo(), block.getTArguments()->data(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size());
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template <typename T>
        LegacyReduce3Op<T>::LegacyReduce3Op() : LegacyOp<T>::LegacyOp(2) {
            //
        }

        template <typename T>
        LegacyReduce3Op<T>::LegacyReduce3Op(int opNum) : LegacyOp<T>::LegacyOp(2, opNum) {
            //
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        ShapeList *LegacyReduce3Op<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            auto xShape = inputShape->at(0);
            auto yShape = inputShape->at(1);

            int *zShape = nullptr;

            if (shape::equalsSoft(xShape, yShape) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && block.getIArguments()->at(0) == MAX_INT))) {
                // reduce3 to scalar case
                ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
                zShape[0] = 2;
                zShape[1] = 1;
                zShape[2] = 1;
                zShape[3] = 1;
                zShape[4] = 1;
                zShape[5] = 0;
                zShape[6] = 1;
                zShape[7] = 99;
            } else {
                auto array = new NDArray<T>(nullptr, xShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                xShape = ShapeUtils<T>::evalReduceShapeInfo('c', *block.getIArguments(), *array);

                delete array;
            }

            return new ShapeList(zShape);
        }

        template class ND4J_EXPORT LegacyReduce3Op<float>;
        template class ND4J_EXPORT LegacyReduce3Op<float16>;
        template class ND4J_EXPORT LegacyReduce3Op<double>;
    }
}