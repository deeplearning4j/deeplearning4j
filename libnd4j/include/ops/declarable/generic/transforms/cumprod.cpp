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
#if NOT_EXCLUDED(OP_cumprod)

#include <ops/declarable/helpers/prefix.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(cumprod, 1, 1, true, 0, 2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            const bool exclusive = INT_ARG(0) == 1;
            const bool reverse = INT_ARG(1) == 1;

            if (block.getIArguments()->size() == 2 && block.width() == 1) {
                // all at once case
                nd4j::ops::helpers::_prefix<T, simdOps::Multiply<T>>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo(), exclusive, reverse);
            } else {
                std::vector<int> dims(block.numI() - 2);

                if (block.width() == 1) {

                    for (int e = 0; e < block.numI() - 2; e++)
                        dims[e] = INT_ARG(e + 2);
                } else {
                    auto ax = INPUT_VARIABLE(1);
                    dims = ax->template asVectorT<int>();
                }

                for (int e = 0; e < dims.size(); e++)
                    if (dims[e] < 0)
                        dims[e] += input->rankOf();

                nd4j::ops::helpers::_prefix<T, simdOps::Multiply<T>>(input, output, dims, exclusive, reverse);
            }

            return ND4J_STATUS_OK;
        }

        CUSTOM_OP_IMPL(cumprod_bp, 2, -1, false, 0, 2) {
            auto input = INPUT_VARIABLE(0);
            auto axis = block.width() == 3 ? INPUT_VARIABLE(1) : nullptr;
            auto gradOut = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
    
            const bool exclusive = INT_ARG(0) == 1;
            const bool reverse = INT_ARG(1) == 1;
    
            std::vector<int> dims;
    
            if (block.width() > 2) {
                dims = axis->template asVectorT<int>();
                OUTPUT_VARIABLE(1)->assign(static_cast<T>(1.0f));
            } else if (int newSize = (block.numI() - 2)) {
                dims.resize(newSize);
    
                for (int e = 0; e < newSize; e++)
                    dims[e] = INT_ARG(e + 2);
            }

            nd4j::ops::helpers::_prefix<T, simdOps::Multiply<T>>(input, output, dims, exclusive, reverse);
            std::unique_ptr<NDArray<T>> val(output->dup());
 
            gradOut->template applyPairwiseTransform<simdOps::Multiply<T>>(output, val.get(), nullptr);
            val->template applyPairwiseTransform<simdOps::Divide<T>>(input, val.get(), nullptr);
            if (!exclusive && !reverse) {
                if (dims.size())
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val.get(), output, dims, false, true);
                else
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), false, true);
    
            }
            else if (!exclusive && reverse){
                if (dims.size())
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val.get(), output, dims, false, false);
                else
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), false, false);
            }
            else if (exclusive && !reverse) {
                if (dims.size())
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val.get(), output, dims, true, true);
                else
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), true, true);
            }
            else {
                if (dims.size())
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val.get(), output, dims, true, false);
                else
                    nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(val->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), true, false);
            }
                
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(cumprod_bp) {
            auto inp = inputShape->at(0);
            Nd4jLong *newShapeX = nullptr;
            COPY_SHAPE(inp, newShapeX);

            if (block.width() == 2) {
                return SHAPELIST(newShapeX);
            } else {
                Nd4jLong *newShapeA = nullptr;
                COPY_SHAPE(inputShape->at(1), newShapeA);

                return SHAPELIST(newShapeX, newShapeA);
            }
        }
    }
}

#endif