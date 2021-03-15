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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(relu_layer, 3, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto w = INPUT_VARIABLE(1);
            auto b = INPUT_VARIABLE(2);

            REQUIRE_TRUE(x->isMatrix(), 0, "relu_layer: x argument should be a 2D tensor, but got rank %i instead!", x->rankOf());
            REQUIRE_TRUE(w->isMatrix(), 0, "relu_layer: weights argument should be a 2D tensor, but got rank %i instead!", w->rankOf());
            REQUIRE_TRUE(b->isVector(), 0, "relu_layer: biases argument should be a 1D tensor, but got rank %i instead!", b->rankOf());
            REQUIRE_TRUE(b->lengthOf() == w->sizeAt(1), 0, "relu_layer: biases array length should match to columns of weights matrix, however got length = %i and columns = %i!", b->lengthOf(), w->sizeAt(1));
            REQUIRE_TRUE(x->sizeAt(1) == w->sizeAt(0), 0, "relu_layer: number of x columns should match to row number of weights matrix, but got x_columns = %i and weights_rows = %i!", x->sizeAt(1), w->sizeAt(0));

            auto output = OUTPUT_VARIABLE(0);

            sd::ops::xw_plus_b op;
            auto status = op.execute({x, w, b}, {output});
            REQUIRE_TRUE(Status::OK() == status, 0, "relu_layer: xw_plus_b op failed on input data.");

            auto scalar = block.numT() > 0 ? block.getTArguments()->at(0) : 0.0;

            output->applyScalar(sd::scalar::RELU, scalar, *output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(relu_layer) {
            auto inShape = inputShape->at(0);
            auto weightsShape = inputShape->at(1);
            auto outputShape = ShapeUtils::matrixProductShape(inShape, weightsShape, false, false, ArrayOptions::dataType(inShape), block.getWorkspace());

            return SHAPELIST(outputShape);
        }

        DECLARE_TYPES(relu_layer) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
//                  ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

