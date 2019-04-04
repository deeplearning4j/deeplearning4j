/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Paul Dubs
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_dot_product_attention)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>


namespace nd4j {
namespace ops  {

    CUSTOM_OP_IMPL(dot_product_attention, 3, -1, false, 0, 2) {
        auto queries = INPUT_VARIABLE(0);
        auto keys = INPUT_VARIABLE(1);
        auto values = INPUT_VARIABLE(2);
        auto mask    = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

        auto output = OUTPUT_VARIABLE(0);
        NDArray* weights;
        bool outputWeights = INT_ARG(1);
        if(outputWeights){
            weights = OUTPUT_VARIABLE(1);
        }else{
            auto weightShape = ShapeUtils::evalShapeForMatmul(keys->getShapeInfo(), queries->getShapeInfo(), true, false);
            weights = NDArrayFactory::create_('c', weightShape, values->dataType(), block.workspace());
        }

        int normalization = INT_ARG(0);

        REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
                     "dot_product_attention: Queries, Keys and Values must have same rank. "
                     "But got queries = %s, keys = %s, values = %s", ShapeUtils::shapeAsString(queries).c_str(),
                     ShapeUtils::shapeAsString(keys).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
                     "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
                     "or rank 4 arrays for multi headed attention. But got rank = %i", queries->rankOf());

        REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
                "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
                "But got queries = %i, keys = %i, values = %i", queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

        REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
                "dot_product_attention: Queries and Keys must have the same feature size. "
                "But got queries = %i, keys = %i", queries->sizeAt(-2), keys->sizeAt(-2));

        REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
                "dot_product_attention: Keys and Values must have the same timestep length. "
                "But got keys = %i, values = %i", keys->sizeAt(-1), values->sizeAt(-1));

        nd4j::ops::matmul mmul;
        mmul.execute({keys, queries}, {weights}, {}, {1}, {});
        if(normalization) {
            *weights /= sqrt((double)keys->sizeAt(-2));
        }

        if(mask != nullptr){
            NDArray* reshapedMask;
            if(weights->rankOf() == 4){
                reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), 1, mask->sizeAt(1), 1});
            }else{
                reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(1), 1});
            }
            *weights += (*reshapedMask - 1) * 1e9;
            delete reshapedMask;
        }

        nd4j::ops::softmax softmax;
        softmax.execute({weights}, {weights}, {}, {-2}, {}, true);

        mmul.execute({values, weights}, {output}, {}, {}, {});

        if(!outputWeights){
            delete weights;
        }

        return Status::OK();
    }


    DECLARE_TYPES(dot_product_attention) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(dot_product_attention) {
        auto query_shape = inputShape->at(0);
        auto keys_shape = inputShape->at(1);
        auto values_shape = inputShape->at(2);

        Nd4jLong *weights_shape = ShapeBuilders::createShapeInfo(nd4j::ArrayOptions::dataType(values_shape), 'c',
                                                                 ShapeUtils::evalShapeForMatmul(keys_shape, query_shape, true, false), block.workspace());
        Nd4jLong *output_shape = ShapeBuilders::createShapeInfo(nd4j::ArrayOptions::dataType(values_shape), 'c',
                                                                ShapeUtils::evalShapeForMatmul(values_shape, weights_shape, false, false), block.workspace());

        if(INT_ARG(1)){
            return SHAPELIST(output_shape, weights_shape);
        }else{
            return SHAPELIST(output_shape);
        }

    }

    CUSTOM_OP_IMPL(dot_product_attention_bp, 4, 3, false, 0, 1) {
        auto queries = INPUT_VARIABLE(0);
        auto keys = INPUT_VARIABLE(1);
        auto values = INPUT_VARIABLE(2);
        auto eps = INPUT_VARIABLE(3);
        auto mask    = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

        auto dLdq = OUTPUT_VARIABLE(0);
        auto dLdk = OUTPUT_VARIABLE(1);
        auto dLdv = OUTPUT_VARIABLE(2);

        int normalization = INT_ARG(0);


       REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
                     "dot_product_attention: Queries, Keys and Values must have same rank. "
                     "But got queries = %s, keys = %s, values = %s", ShapeUtils::shapeAsString(queries).c_str(),
                     ShapeUtils::shapeAsString(keys).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
                     "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
                     "or rank 4 arrays for multi headed attention. But got rank = %i", queries->rankOf());

        REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
                     "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
                     "But got queries = %i, keys = %i, values = %i", queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

        REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
                     "dot_product_attention: Queries and Keys must have the same feature size. "
                     "But got queries = %i, keys = %i", queries->sizeAt(-2), keys->sizeAt(-2));

        REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
                     "dot_product_attention: Keys and Values must have the same timestep length. "
                     "But got keys = %i, values = %i", keys->sizeAt(-1), values->sizeAt(-1));


        double factor;
        if(normalization)
            factor = sqrt((double)keys->sizeAt(-2));

        nd4j::ops::matmul mmul;
        auto preSoftmaxRes = mmul.execute({keys, queries}, {}, {1}, {});
        auto preSoftmax = preSoftmaxRes->at(0);
        if(normalization)
            *preSoftmax /= factor;

        if(mask != nullptr){
            NDArray* reshapedMask;
            if(preSoftmax->rankOf() == 4){
                reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), 1, mask->sizeAt(1), 1});
            }else{
                reshapedMask = mask->reshape(mask->ordering(), {mask->sizeAt(0), mask->sizeAt(1), 1});
            }
            *preSoftmax += (*reshapedMask - 1) * 1e9;
            delete reshapedMask;
        }

        nd4j::ops::softmax softmax;
        auto weightsRes = softmax.execute({preSoftmax}, {}, {-2}, {});
        auto weights = weightsRes->at(0);

        nd4j::ops::matmul_bp mmul_bp;
        NDArray dLdw(weights->getShapeInfo(), block.workspace());
        mmul_bp.execute({values, weights, eps}, {dLdv, &dLdw}, {}, {}, {});

        nd4j::ops::softmax_bp softmax_bp;
        auto dLdsRes = softmax_bp.execute({preSoftmax, &dLdw}, {}, {-2}, {});
        auto dLds = dLdsRes->at(0);

        if(normalization)
            *dLds /= factor;

        mmul_bp.execute({keys, queries, dLds}, {dLdk, dLdq}, {}, {1}, {});

        delete preSoftmaxRes;
        delete weightsRes;
        delete dLdsRes;

        return Status::OK();
    }

    DECLARE_TYPES(dot_product_attention_bp) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(dot_product_attention_bp) {
        Nd4jLong *dLdq_shape;
        COPY_SHAPE(inputShape->at(0), dLdq_shape);
        Nd4jLong *dLdk_shape;
        COPY_SHAPE(inputShape->at(1), dLdk_shape);
        Nd4jLong *dLdv_shape;
        COPY_SHAPE(inputShape->at(2), dLdv_shape);

        return SHAPELIST(dLdq_shape, dLdk_shape, dLdv_shape);
    }

}
}

#endif