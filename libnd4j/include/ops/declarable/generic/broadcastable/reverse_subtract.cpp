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
#if NOT_EXCLUDED(OP_reversesubtract)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        BROADCASTABLE_OP_IMPL(reversesubtract, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper::broadcastApply(BROADCAST(ReverseSubtract), x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

			return Status::OK();
        }
        DECLARE_SYN(RSub, reversesubtract);


        CUSTOM_OP_IMPL(reversesubtract_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            if (x->isSameShape(y)) {
                // PWT case case
                epsNext->applyTransform(transform::Neg, gradX, nullptr);
                gradY->assign(epsNext);
            } else if (y->isScalar()) {
                // scalar case
                auto tmp = epsNext->reduceNumber(reduce::Sum);
                gradY->assign(tmp);
                epsNext->applyTransform(transform::Neg, gradX, nullptr);
            } else {
                // broadcastable
                auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = epsNext->reduceAlongDimension(reduce::Sum, axisX);
                    sum->applyTransform(transform::Neg, gradX);
                    delete sum;
                } else {
                    epsNext->applyTransform(transform::Neg, gradX, nullptr);
                }

                if (axisY.size() > 0) {
                    auto sum = epsNext->reduceAlongDimension(reduce::Sum, axisY);
                    gradY->assign(sum);
                    delete sum;
                } else {
                    gradY->assign(epsNext);
                }
            }  

            return Status::OK();
        }

        DECLARE_SHAPE_FN(reversesubtract_bp) {
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