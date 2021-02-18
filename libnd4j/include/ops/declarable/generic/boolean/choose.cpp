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
//  @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_choose)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/choose.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 2, false, -2, -1) {

            int mode = INT_ARG(0);
            auto result = OUTPUT_VARIABLE(0);
            auto numResults = OUTPUT_VARIABLE(1);

            if (block.width() > 1) {
                auto arg = INPUT_VARIABLE(0);
                auto comp = INPUT_VARIABLE(1);

                helpers::chooseFunctorArray(block.launchContext(), arg, comp, mode, result, numResults);

            }//scalar case
            else {
                double scalar = T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                helpers::chooseFunctorScalar(block.launchContext(), arg, scalar, mode, result, numResults);
            }


            return Status::OK();
        }

        DECLARE_TYPES(choose) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_INTS});
        }

        DECLARE_SHAPE_FN(choose) {
            Nd4jLong const* shape;
            int rank;
            int mode = INT_ARG(0);
            auto numResults = NDArrayFactory::create<Nd4jLong>(0L);
            if(block.width() > 1) {
                auto first = INPUT_VARIABLE(0);
                auto second = INPUT_VARIABLE(1);
                if(first->lengthOf() > second->lengthOf()) {
                    shape = first->shapeInfo();
                    rank = first->rankOf();
                }
                else {
                    shape = second->shapeInfo();
                    rank = second->rankOf();
                }

                helpers::chooseFunctorArray(block.launchContext(), first, second, mode, nullptr, &numResults);
            }
            else {
                auto first = INPUT_VARIABLE(0);
                shape = first->shapeInfo();
                rank = first->rankOf();
                double scalar = T_ARG(0);

                helpers::chooseFunctorScalar(block.launchContext(), first, scalar, mode, nullptr, &numResults);
            }

            auto newShape = ConstantShapeHelper::getInstance().vectorShapeInfo(numResults.e<Nd4jLong>(0), ArrayOptions::dataType(inputShape->at(0)));

            auto shapeScalar = ConstantShapeHelper::getInstance().scalarShapeInfo(sd::DataType::INT64);
            return SHAPELIST(newShape, shapeScalar);
        }


    }
}

#endif