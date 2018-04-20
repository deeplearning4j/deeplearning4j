//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_realdiv)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(realdiv, 2, 1, true, 0, 0) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);


            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Divide<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            /*
            if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                // REQUIRE_OK(this->validateInputDimensionsMatch(block));
                x->template applyPairwiseTransform<simdOps::Divide<T>>(y, z, nullptr);

            } else if (!x->isScalar() && y->isScalar()) {
                x->template applyScalar<simdOps::Divide<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Divide<T>>(*x, z);
            }
            else if (x->isScalar() && y->isScalar()) { // (x->isScalar() && y->isScalar())
                z->putScalar(0, x->getScalar(0) / y->getScalar(0));
            } else if (ShapeUtils<T>::areShapesBroadcastable(*x, *y)){
                auto tZ = x->template applyTrueBroadcast<simdOps::Divide<T>>(y);
                OVERWRITE_RESULT(tZ);
            } else {
                auto sx = ShapeUtils<T>::shapeAsString(*x);
                auto sy = ShapeUtils<T>::shapeAsString(*y);
                REQUIRE_TRUE(false, 0, "RealDiv: shapes should be equal, or broadcastable. But got %s vs %s instead", sx.c_str(), sy.c_str());
            }
            */

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RealDiv, realdiv);

        DECLARE_SHAPE_FN(realdiv) {
            auto shapeList = SHAPELIST();
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
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

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


        CUSTOM_OP_IMPL(realdiv_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            auto lambdaX = LAMBDA_TT(_e, _y) {
                return _e / _y;
            };

            auto lambdaY = LAMBDA_TTT(_e, _x, _y) {
                return _e * -_x / (_y * _y);
            };


            if (x->isSameShape(y)) {
                // PWT case case

                // X gradient
                epsNext->applyPairwiseLambda(y, lambdaX, gradX);

                // Y gradient
                epsNext->applyTriplewiseLambda(x, y, lambdaY, gradY);

            } else if (y->isScalar()) {
                // scalar case
                T _y = y->getScalar(0);
                auto lambdaS = LAMBDA_T(_e, _y) {
                    return _e / _y;
                };

                T tmp = epsNext->template reduceNumber<simdOps::Sum<T>>();
                T tmpX = x->template reduceNumber<simdOps::Sum<T>>();
                gradY->assign(tmp * -tmpX / (_y * _y));
                
                epsNext->applyLambda(lambdaS, gradX);
            } else {
                // broadcast case

                auto preX = *epsNext / *y;

                NDArray<T> negX(*x);
                x->template applyTransform<simdOps::Neg<T>>(&negX);
                auto preY = *epsNext * negX / ((*y) * (*y));

                auto axisX = ShapeUtils<T>::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils<T>::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = preX.template reduceAlongDimension<simdOps::Sum<T>>(axisX);
                    gradX->assign(sum);
                    delete sum;
                } else 
                    gradX->assign(preX);

                if (axisY.size() > 0) {
                    auto sum = preY.template reduceAlongDimension<simdOps::Sum<T>>(axisY);
                    gradY->assign(sum);
                    delete sum;
                } else
                    gradY->assign(preY);
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(realdiv_bp) {
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);
            auto e = inputShape->at(2);

            // eps always has shape of x
            // grad always has shape of y

            int *shapeE;
            int *shapeG;
            ALLOCATE(shapeE, block.getWorkspace(), shape::shapeInfoLength(x), int);
            ALLOCATE(shapeG, block.getWorkspace(), shape::shapeInfoLength(y), int);

            REPLICATE_SHAPE(x, shapeE);
            REPLICATE_SHAPE(y, shapeG);

            auto shapeList = SHAPELIST(shapeE, shapeG);

            return shapeList;
        }
    }
}

#endif