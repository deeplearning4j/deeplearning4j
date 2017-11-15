//
// Created by raver119 on 16.10.2017.
//

#include "ops/declarable/LegacyReduceOp.h"
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp() : LegacyOp<T>::LegacyOp(1) {
            //
        }

        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //this->_opNum = opNum;
        }

        template <typename T>
        Nd4jStatus LegacyReduceOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
            bool allAxes = false;

            if (block.getIArguments()->size() == x->rankOf())
                allAxes = true;

            if ((block.getIArguments()->size() == 0) ||
                (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                // scalar
                T res = NativeOpExcutioner<T>::execReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data());
                z->putScalar(0, res);
            } else {
                // TAD
                std::vector<int> dims(*block.getIArguments());
                std::sort(dims.begin(), dims.end());

                REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

                shape::TAD tad(x->getShapeInfo(), dims.data(), dims.size());
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                NativeOpExcutioner<T>::execReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo(), dims.data(), (int) dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        ShapeList *LegacyReduceOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;

            bool allAxes = false;

            if (block.getIArguments()->size() == shape::rank(inShape))
                allAxes = true;

            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
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


        template class ND4J_EXPORT LegacyReduceOp<float>;
        template class ND4J_EXPORT LegacyReduceOp<float16>;
        template class ND4J_EXPORT LegacyReduceOp<double>;
    }
}