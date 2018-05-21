//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_assign)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(assign, 2, 1, true, 0, 0) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Copy<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }
    
			return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(assign) {
            auto shapeList = SHAPELIST();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);

            if (shape::equalsSoft(x, y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(y, newshape);

                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (ShapeUtils<T>::areShapesBroadcastable(x, y)) {
                Nd4jLong *newshape = nullptr;
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        }
        DECLARE_SYN(set, assign);
        DECLARE_SYN(copy, assign);

        CUSTOM_OP_IMPL(assign_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            gradX->assign((T) 0.0f);

            if (x->isSameShape(y)) {
                gradY->assign(epsNext);
            } else if (y->isScalar()) {
                T sum = epsNext->template reduceNumber<simdOps::Sum<T>>();
                gradY->assign(sum);
            } else {
                // broadcastable
                auto axisY = ShapeUtils<T>::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisY.size() > 0) {
                    auto sum = epsNext->template reduceAlongDimension<simdOps::Sum<T>>(axisY);
                    gradY->assign(sum);
                    delete sum;
                } else
                    gradY->assign(epsNext);
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(assign_bp) {
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);
            auto e = inputShape->at(2);

            // eps always has shape of x
            // grad always has shape of y

            Nd4jLong *shapeE;
            Nd4jLong *shapeG;
            
            COPY_SHAPE(x, shapeE);
            COPY_SHAPE(y, shapeG);

            auto shapeList = SHAPELIST(shapeE, shapeG);

            return shapeList;
        }
    }
}

#endif