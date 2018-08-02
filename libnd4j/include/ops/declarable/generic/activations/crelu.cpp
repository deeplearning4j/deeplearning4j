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
#if NOT_EXCLUDED(OP_crelu)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(crelu, 1, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);

            auto tmp = x->dup();
            tmp->template applyTransform<simdOps::Neg<T>>();

            auto z = OUTPUT_VARIABLE(0);

            helpers::concat({x, tmp}, *z, x->rankOf()-1);
            // NDArrayFactory<T>::concat({x, tmp}, -1, z);

            // TODO: make this configurable?
            T threshold = (T) 0.0f;
            z->template applyTransform<simdOps::RELU<T>>(&threshold);

            STORE_RESULT(z);

            delete tmp;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(crelu) {
            auto inShape = inputShape->at(0);
            std::vector<Nd4jLong> shape;
            for (int e = 0; e < shape::rank(inShape); e++)
                shape.emplace_back(shape::shapeOf(inShape)[e]);
            
            shape[shape.size()-1] *= 2;
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
            if (shape::order(inShape) == 'c')
                shape::shapeBuffer(shape.size(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }


        CUSTOM_OP_IMPL(crelu_bp, 2, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto epsilonNext = INPUT_VARIABLE(1);
            auto epsilon = OUTPUT_VARIABLE(0);

            // at first step we build fwd activation
            nd4j::ops::crelu<T> op;
            auto tmpResult = op.execute({input}, {}, {}); 
            if (tmpResult->status() != ND4J_STATUS_OK)
                return tmpResult->status();

            auto actv = tmpResult->at(0);

            // now we do RELU backward pass
            auto lambda = LAMBDA_TT(_x, _e) {
                return _x > (T) 0.0f ? _e  : (T) 0.0f;
            };
            actv->applyPairwiseLambda(epsilonNext, lambda);

            // now we split updated array into 2 chunks along last dimension
            nd4j::ops::concat_bp<T> opc;
            auto dec = opc.execute({input, input, actv}, {},{-1});
            if (dec->status() != ND4J_STATUS_OK)
                return dec->status();

            // and now we subtract two parts of epsilons and pass result out
            auto pos = dec->at(0);
            auto neg = dec->at(1);

            pos->template applyPairwiseTransform<simdOps::Subtract<T>>(neg, epsilon, nullptr);

            delete tmpResult;
            delete dec;
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(crelu_bp) {
            auto inShape = inputShape->at(0);
            Nd4jLong* newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif