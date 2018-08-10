/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 12.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_minimum)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace nd4j {
    namespace ops {
        BROADCASTABLE_OP_IMPL(minimum, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcastApply<simdOps::Min<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            return ND4J_STATUS_OK;
        }

        CUSTOM_OP_IMPL(minimum_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            auto lambdaX = LAMBDA_TTT(_e, _x, _y) {
                return _x <= _y ? _e : (T) 0.;
            };

            auto lambdaY = LAMBDA_TTT(_e, _x, _y) {
                return _x >= _y ? _e : (T) 0.;
            };


            if (x->isSameShape(y)) {
                // PWT case case

                // X gradient
                epsNext->applyTriplewiseLambda(x, y, lambdaX, gradX);

                // Y gradient
                epsNext->applyTriplewiseLambda(x, y, lambdaY, gradY);

            } else if (y->isScalar()) {
                T s = y->getScalar(0);
                auto lambdaS = LAMBDA_TT(_e, _x, s) {
                    return _x <= s ? _e : (T) 0.;
                };

                // scalar case
                T tmp = epsNext->template reduceNumber<simdOps::Sum<T>>();
                gradY->assign( x <= y ? tmp : (T) 0.0f);
                
                epsNext->applyPairwiseLambda(x, lambdaS, gradX);
            } else {
                // broadcast case

                // in this case we want to boost our X and Y shapes to the size of FF pass output (or epsNext, which has the same shape)
                auto preX = x->dup();
                auto preY = y->dup();

                auto targetShape = epsNext->getShapeAsVector();

                preX->tileToShape(targetShape);
                preY->tileToShape(targetShape);

                epsNext->applyTriplewiseLambda(preX, preY, lambdaX, preX);
                epsNext->applyTriplewiseLambda(preX, preY, lambdaY, preY);

                auto axisX = ShapeUtils<T>::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils<T>::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = preX->template reduceAlongDimension<simdOps::Sum<T>>(axisX);
                    gradX->assign(sum);
                    delete sum;
                } else 
                    gradX->assign(preX);

                if (axisY.size() > 0) {
                    auto sum = preY->template reduceAlongDimension<simdOps::Sum<T>>(axisY);
                    gradY->assign(sum);
                    delete sum;
                } else
                    gradY->assign(preY);


                delete preX;
                delete preY;
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(minimum_bp) {
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