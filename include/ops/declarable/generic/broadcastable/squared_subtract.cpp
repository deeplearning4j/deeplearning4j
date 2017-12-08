//
// Created by raver119 on 23.11.17.
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(squaredsubtract, 2, 1, true, 0, 0) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);


            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::SquaredSubtract<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            /*
            if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                x->template applyPairwiseTransform<simdOps::SquaredSubtract<T>>(y, z, nullptr);

            } else if (!x->isScalar() && y->isScalar()) {
                x->template applyScalar<simdOps::SquaredSubtract<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::SquaredSubtract<T>>(*x, z);
            }
            else if (x->isScalar() && y->isScalar()) { // x->isScalar() && y->isScalar()
                z->putScalar(0, nd4j::math::nd4j_pow(x->getScalar(0) - y->getScalar(0), (T) 2));
            } else if (ShapeUtils<T>::areShapesBroadcastable(*x, *y)) {
                auto tZ = x->template applyTrueBroadcast<simdOps::SquaredSubtract<T>>(y);
                OVERWRITE_RESULT(tZ);
            } else {
                auto sx = ShapeUtils<T>::shapeAsString(*x);
                auto sy = ShapeUtils<T>::shapeAsString(*y);
                REQUIRE_TRUE(false, 0, "SquaredSubtract: shapes should be equal, or broadcastable. But got %s vs %s instead", sx.c_str(), sy.c_str());
            }
            */

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(squareddifference, squaredsubtract);

        DECLARE_SHAPE_FN(squaredsubtract) {
            auto shapeList = new ShapeList();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);

            if (shape::equalsSoft(x, y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(y), int);
                REPLICATE_SHAPE(y, newshape);

                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (ShapeUtils<T>::areShapesBroadcastable(x, y)) {
                int *newshape = nullptr;
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape);

                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        }
    }
}