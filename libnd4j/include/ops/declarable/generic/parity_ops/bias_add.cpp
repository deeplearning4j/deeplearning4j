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
#if NOT_EXCLUDED(OP_biasadd)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(biasadd, 2, 1, true) {
            //REQUIRE_OK(this->validateInput2D(block));
            auto input = INPUT_VARIABLE(0);
            auto bias = INPUT_VARIABLE(1);

            REQUIRE_TRUE(bias->isRowVector(), 0, "Bias array should be a vector");

            auto z = OUTPUT_VARIABLE(0);

            if (input->isMatrix())
                input->addRowVector(bias, z);
            else {
                // TODO: we might want to use NDArray::applyTrueBroadcast here, like AddOp does
                std::vector<Nd4jLong> shape({-1, bias->lengthOf()});
                //nd4j_debug("Reshaping to: [%i, %i]\n", -1, (int) bias->lengthOf());
                auto tArr = input->reshape(input->ordering(), shape);
                auto zArr = z->reshape(z->ordering(), shape);
                tArr->addRowVector(bias, zArr);

                delete tArr;
                delete zArr;
            }

            STORE_RESULT(*z);

            return Status::OK();
        }
        DECLARE_SYN(bias_add, biasadd);


        CUSTOM_OP_IMPL(biasadd_bp, 3, 2, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto bias = INPUT_VARIABLE(1);
            auto epsilonNext = INPUT_VARIABLE(2);

            auto epsilon = OUTPUT_VARIABLE(0);
            auto gradB = OUTPUT_VARIABLE(1);

            epsilon->assign(epsilonNext);

            // cnn case
            if (input->rankOf() == 4) {
                auto epsilonNext2d = epsilonNext->permute({1, 0, 2, 3});
                epsilonNext2d->reshapei('c', {(int) bias->lengthOf(), -1});

                auto sum = epsilonNext2d->reduceAlongDimension(reduce::Sum, {1});
                gradB->assign(sum);

                delete sum;
                delete epsilonNext2d;
            } else if (input->rankOf() == 2) {
                // regular fully-connected case
                auto sum = epsilonNext->reduceAlongDimension(reduce::Sum, {0});
                gradB->assign(sum);
                
                delete sum;
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(BiasAddGrad, biasadd_bp);

        DECLARE_SHAPE_FN(biasadd_bp) {
            auto input = inputShape->at(0);
            auto bias = inputShape->at(1);

            Nd4jLong* epsShape;
            Nd4jLong* gradShape;

            COPY_SHAPE(input, epsShape);
            COPY_SHAPE(bias, gradShape);

            return SHAPELIST(epsShape, gradShape);
        }
    }
}

#endif