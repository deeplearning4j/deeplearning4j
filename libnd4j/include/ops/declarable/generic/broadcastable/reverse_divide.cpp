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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reversedivide)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        BROADCASTABLE_OP_IMPL(reversedivide, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            // auto tZ = BroadcastHelper<T>::template broadcastApply<simdOps::ReverseDivide<T>>(x, y, z);
            x->applyTrueBroadcast(BROADCAST(ReverseDivide), y, z, true);
            // if (tZ == nullptr)
            //     return ND4J_STATUS_KERNEL_FAILURE;
            // else if (tZ != z) {
            //     OVERWRITE_RESULT(tZ);
            // }

			return Status::OK();
        }
        DECLARE_SYN(RDiv, reversedivide);

        CUSTOM_OP_IMPL(reversedivide_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            if (x->isSameShape(y)) {
                // PWT case case

                // X gradient
                //epsNext->applyTriplewiseLambda(x, y, lambdaX, gradX);
                gradX->assign(epsNext * -(*y) / ((*x) * (*x)));

                // Y gradient
                //epsNext->applyPairwiseLambda(x, lambdaY, gradY);
                epsNext->applyPairwiseTransform(pairwise::Divide, x, gradY, nullptr);

            } else if (y->isScalar()) {
                // scalar case
                auto tmp = epsNext->reduceNumber(reduce::Sum);
                auto tmpX = x->reduceNumber(reduce::Sum);
                gradY->assign(tmp / tmpX);

                gradX->assign(epsNext * -(*y) / ((*x) * (*x)));
            } else {
                // broadcast case

                auto preY = (*epsNext) / (*x);

                NDArray negY(*y);
                y->applyTransform(transform::Neg, &negY);
                auto preX = *epsNext * negY / ((*x) * (*x));

                auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = preX.reduceAlongDimension(reduce::Sum, axisX);
                    gradX->assign(sum);
                    delete sum;
                } else 
                    gradX->assign(preX);

                if (axisY.size() > 0) {
                    auto sum = preY.reduceAlongDimension(reduce::Sum, axisY);
                    gradY->assign(sum);
                    delete sum;
                } else
                    gradY->assign(preY);
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(reversedivide_bp) {
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