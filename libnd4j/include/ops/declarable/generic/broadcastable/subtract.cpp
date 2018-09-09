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
#if NOT_EXCLUDED(OP_subtract)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        BROADCASTABLE_OP_IMPL(subtract, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Subtract(), x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

			return Status::OK();
        }
        DECLARE_SYN(Sub, subtract);
        DECLARE_SYN(sub, subtract);

        CUSTOM_OP_IMPL(subtract_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            if (x->isSameShape(y)) {
                // PWT case case
                epsNext->template applyTransform<simdOps::Neg<T>>(gradY, nullptr);
                gradX->assign(epsNext);
            } else if (y->isScalar()) {
                // scalar case
                auto tmp = epsNext->template reduceNumber<simdOps::Sum<T>>();
                gradY->assign(-tmp);
                gradX->assign(epsNext);
            } else {
                // broadcastable
                auto axisX = ShapeUtils<T>::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils<T>::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = epsNext->template reduceAlongDimension<simdOps::Sum<T>>(axisX);
                    gradX->assign(sum);
                    delete sum;
                } else 
                    gradX->assign(epsNext);

                if (axisY.size() > 0) {
                    auto sum = epsNext->template reduceAlongDimension<simdOps::Sum<T>>(axisY);
                    sum->template applyTransform<simdOps::Neg<T>>(gradY);
                    delete sum;
                } else {
                    epsNext->template applyTransform<simdOps::Neg<T>>(gradY);
                }
            }  

            return Status::OK();
        }

        DECLARE_SHAPE_FN(subtract_bp) {
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