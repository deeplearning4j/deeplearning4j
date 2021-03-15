/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reversedivide)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        BROADCASTABLE_OP_IMPL(reversedivide, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            BROADCAST_CHECK_EMPTY(x,y,z);

            REQUIRE_TRUE(!x->isB(), 0, "REVERSEDIVIDE OP: you can't divide by bool array!");
            x->applyTrueBroadcast(BROADCAST(ReverseDivide), *y, *z, true);

			return Status::OK();
        }
        DECLARE_SYN(RDiv, reversedivide);

        DECLARE_TYPES(reversedivide) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedInputTypes(1, DataType::ANY)
                    ->setAllowedOutputTypes(0, DataType::INHERIT);
        }

        DECLARE_TYPES(reversedivide_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

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
                gradX->assign((*epsNext) * (*y) / ((*x) * (*x)));
                gradX->applyTransform(transform::Neg, *gradX);
                // Y gradient
                //epsNext->applyPairwiseLambda(x, lambdaY, gradY);
                gradY->assign((*epsNext) / (*x));
            } else if (y->isScalar()) {
                // scalar case
                auto tmp = epsNext->reduceNumber(reduce::Sum);
                auto tmpX = x->reduceNumber(reduce::Sum);
                gradY->assign(tmp / tmpX);

                gradX->assign((*epsNext) * (*y) / ((*x) * (*x)));
                gradX->applyTransform(transform::Neg, *gradX);
            } else {
                // broadcast case

                auto preY = (*epsNext) / (*x);

                auto preX = *epsNext * (*y) / ((*x) * (*x));
                preX.applyTransform(transform::Neg, preX);

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

            return SHAPELIST(CONSTANT(shapeE), CONSTANT(shapeG));
        }
    }
}

#endif