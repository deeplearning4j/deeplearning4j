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
//  @author sgazeos@gmail.com
//

#include <op_boilerplate.h>
#include <NDArray.h>
#include <helpers/ShapeUtils.h>


namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void maximumBPFunctor_(NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {

                auto lambdaX = LAMBDA_TTT(_e, _x, _y) {
                    return _x >= _y ? _e : (T) 0.;
                };

                auto lambdaY = LAMBDA_TTT(_e, _x, _y) {
                    return _x <= _y ? _e : (T) 0.;
                };


                if (x->isSameShape(y)) {
                    // PWT case case

                    // X gradient
                    epsNext->applyTriplewiseLambda(*x, *y, lambdaX, *gradX);

                    // Y gradient
                    epsNext->applyTriplewiseLambda(*x, *y, lambdaY, *gradY);

                } else if (y->isScalar()) {
                    T s = y->e<T>(0);
                    auto lambdaS = LAMBDA_TT(_e, _x, s) {
                        return _x >= s ? _e : (T) 0.;
                    };

                    // scalar case
                    auto tmp = epsNext->reduceNumber(reduce::Sum);
                    if (x <= y)
                        gradY->assign(tmp);
                    else
                        gradY->assign(0.0f);

                    epsNext->applyPairwiseLambda(*x, lambdaS, *gradX);
                } else {
                    // broadcast case

                    // in this case we want to boost our X and Y shapes to the size of FF pass output (or epsNext, which has the same shape)
                    auto preX = x->dup();
                    auto preY = y->dup();

                    auto targetShape = epsNext->getShapeAsVector();

                    preX.tileToShape(targetShape, preX);
                    preY.tileToShape(targetShape, preY);

                    epsNext->applyTriplewiseLambda(preX, preY, lambdaX, preX);
                    epsNext->applyTriplewiseLambda(preX, preY, lambdaY, preY);

                    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                    if (axisX.size() > 0) {
                        auto sum = preX.reduceAlongDimension(reduce::Sum, axisX);
                        gradX->assign(sum);
                    } else
                        gradX->assign(preX);

                    if (axisY.size() > 0) {
                        auto sum = preY.reduceAlongDimension(reduce::Sum, axisY);
                        gradY->assign(sum);
                    } else
                        gradY->assign(preY);
                }
            }

            void maximumBPFunctor(nd4j::LaunchContext * context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
                NDArray::prepareSpecialUse({gradX, gradY}, {x, y, epsNext});

                BUILD_SINGLE_SELECTOR(x->dataType(), maximumBPFunctor_, (x, y, epsNext, gradX, gradY), NUMERIC_TYPES);

                NDArray::registerSpecialUse({gradX, gradY}, {x, y, epsNext});
            }

        }
    }
}