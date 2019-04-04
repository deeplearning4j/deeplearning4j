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
#if NOT_EXCLUDED(OP_multi_head_dot_product_attention)

#include <ops/declarable/CustomOperations.h>
#include <helpers/AttentionHelper.h>

namespace nd4j {
namespace ops  {

    CUSTOM_OP_IMPL(multi_head_dot_product_attention, 7, -1, false, 0, 2) {
        auto queries = INPUT_VARIABLE(0);
        auto keys    = INPUT_VARIABLE(1);
        auto values  = INPUT_VARIABLE(2);
        auto Wq      = INPUT_VARIABLE(3);
        auto Wk      = INPUT_VARIABLE(4);
        auto Wv      = INPUT_VARIABLE(5);
        auto Wo      = INPUT_VARIABLE(6);
        auto mask    = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;


        auto output = OUTPUT_VARIABLE(0);
        int normalization = INT_ARG(0);
        int weights = INT_ARG(1);

        auto numHeads = Wk->sizeAt(0);
        auto miniBatchSize = queries->sizeAt(0);
        auto queryCount = queries->sizeAt(2);
        auto projectedValuesSize = Wv->sizeAt(1);
        auto outSize = Wo->sizeAt(1);

        REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
                     "multi_head_dot_product_attention: Queries, Keys and Values must have same rank. "
                     "But got queries = %s, keys = %s, values = %s", ShapeUtils::shapeAsString(queries).c_str(),
                     ShapeUtils::shapeAsString(keys).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(queries->rankOf() == 3, 0,
                     "multi_head_dot_product_attention: Queries, Keys and Values must be rank 3 arrays"
                     "But got rank = %i", queries->rankOf());

        REQUIRE_TRUE(Wq->rankOf() == Wk->rankOf() && Wk->rankOf() == Wv->rankOf(), 0,
                     "multi_head_dot_product_attention: Input projections weights must have the same rank. "
                     "But got Wq = %s, Wk = %s, Wv = %s", ShapeUtils::shapeAsString(Wq).c_str(),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

        REQUIRE_TRUE(Wq->sizeAt(0) == Wk->sizeAt(0) && Wk->sizeAt(0) == Wv->sizeAt(0), 0,
                     "multi_head_dot_product_attention: Projections weights must have the same number of attention heads. "
                     "But got Wq = %s, Wk = %s, Wv = %s", ShapeUtils::shapeAsString(Wq).c_str(),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

        REQUIRE_TRUE(Wo->rankOf() == 2, 0,
                     "multi_head_dot_product_attention: Output projection weights must have rank 2. "
                     "But got Wo = %s", ShapeUtils::shapeAsString(Wo).c_str());

        REQUIRE_TRUE(Wq->sizeAt(2) == queries->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Query projection matrix Wq has incompatible size to queries matrix."
                     "Expected Wq[2] = queries[1] = %i, but got Wq = %s, queries = %s ", queries->sizeAt(1),
                     ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(queries).c_str());

        REQUIRE_TRUE(Wk->sizeAt(2) == keys->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Key projection matrix Wk has incompatible size to keys matrix."
                     "Expected Wk[2] = keys[1] = %i, but got Wk = %s, keys = %s ", keys->sizeAt(1),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(keys).c_str());

        REQUIRE_TRUE(Wv->sizeAt(2) == values->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Value projection matrix Wv has incompatible size to values matrix."
                     "Expected Wv[2] = values[1] = %i, but got Wv = %s, values = %s ", values->sizeAt(1),
                     ShapeUtils::shapeAsString(Wv).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(Wo->sizeAt(0) == (Wv->sizeAt(1) * Wv->sizeAt(0)), 0,
                     "multi_head_dot_product_attention: Output projection matrix Wo has incompatible size to attention result."
                     "Expected Wo[0] = Wv[0] * Wv[1] = %i, but got Wo = %s, Wv = %", (Wv->sizeAt(1) * Wv->sizeAt(0)),
                     ShapeUtils::shapeAsString(Wo).c_str(), ShapeUtils::shapeAsString(Wv).c_str());


        nd4j::ops::matmul mmul;
        // Project queries, keys, values
        auto projectedQueries = AttentionHelper::multiHeadProject(queries, Wq);
        auto projectedKeys = AttentionHelper::multiHeadProject(keys, Wk);
        auto projectedValues = AttentionHelper::multiHeadProject(values, Wv);

        // Apply Attention
        nd4j::ops::dot_product_attention attention;
        auto results = attention.execute({projectedQueries, projectedKeys, projectedValues, mask}, {}, {normalization, weights}, {});

        if(weights){
            OUTPUT_VARIABLE(1)->assign(results->at(1));
        }

        // Project attention results
        auto attnResults = results->at(0);
        auto attnResultsPerm = attnResults->permute({0, 3, 1, 2});
        auto reshaped = attnResultsPerm->reshape(attnResults->ordering(), {miniBatchSize * queryCount, numHeads * projectedValuesSize});

        auto projRes = mmul.execute({reshaped, Wo}, {}, {}, {});
        auto projReshape = projRes->at(0)->reshape(output->ordering(), {miniBatchSize, queryCount, outSize});
        auto outputProjection = projReshape->permute({0, 2, 1});
        output->assign(outputProjection);

        delete projectedQueries;
        delete projectedKeys;
        delete projectedValues;
        delete results;
        delete attnResultsPerm;
        delete reshaped;
        delete projRes;
        delete projReshape;
        delete outputProjection;

        return Status::OK();
    }


    DECLARE_TYPES(multi_head_dot_product_attention) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(multi_head_dot_product_attention) {
        auto queryShape = inputShape->at(0);
        auto keysShape = inputShape->at(1);
        auto valuesShape = inputShape->at(2);
        auto WkShape = inputShape->at(3);
        auto WoShape = inputShape->at(6);

        auto batchSize = shape::sizeAt(queryShape, 0);
        auto outSize = shape::sizeAt(WoShape, 1);
        auto queryCount = shape::sizeAt(queryShape, 2);
        auto numHeads = shape::sizeAt(WkShape, 0);
        auto timeSteps = shape::sizeAt(keysShape, 2);

        Nd4jLong *weightsShape = ShapeBuilders::createShapeInfo(nd4j::ArrayOptions::dataType(valuesShape), 'c',
                                                                 {batchSize, numHeads, timeSteps, queryCount}, block.workspace());
        Nd4jLong *outputShape = ShapeBuilders::createShapeInfo(nd4j::ArrayOptions::dataType(valuesShape), 'c',
                                                                {batchSize, outSize, queryCount}, block.workspace());

        if(INT_ARG(1)){
            return SHAPELIST(outputShape, weightsShape);
        }else{
            return SHAPELIST(outputShape);
        }

    }

    CUSTOM_OP_IMPL(multi_head_dot_product_attention_bp, 8, 7, false, 0, 1) {
        auto queries = INPUT_VARIABLE(0);
        auto keys    = INPUT_VARIABLE(1);
        auto values  = INPUT_VARIABLE(2);
        auto Wq      = INPUT_VARIABLE(3);
        auto Wk      = INPUT_VARIABLE(4);
        auto Wv      = INPUT_VARIABLE(5);
        auto Wo      = INPUT_VARIABLE(6);
        auto eps     = INPUT_VARIABLE(7);
        auto mask    = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;

        auto dLdq  = OUTPUT_VARIABLE(0);
        auto dLdk  = OUTPUT_VARIABLE(1);
        auto dLdv  = OUTPUT_VARIABLE(2);
        auto dLdWq = OUTPUT_VARIABLE(3);
        auto dLdWk = OUTPUT_VARIABLE(4);
        auto dLdWv = OUTPUT_VARIABLE(5);
        auto dLdWo = OUTPUT_VARIABLE(6);

        int normalization = INT_ARG(0);

        auto numHeads = Wk->sizeAt(0);
        auto miniBatchSize = queries->sizeAt(0);
        auto queryCount = queries->sizeAt(2);
        auto outSize = Wo->sizeAt(1);
        auto projectedValuesSize = Wv->sizeAt(1);


        REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
                     "multi_head_dot_product_attention: Queries, Keys and Values must have same rank. "
                     "But got queries = %s, keys = %s, values = %s", ShapeUtils::shapeAsString(queries).c_str(),
                     ShapeUtils::shapeAsString(keys).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(queries->rankOf() == 3, 0,
                     "multi_head_dot_product_attention: Queries, Keys and Values must be rank 3 arrays"
                     "But got rank = %i", queries->rankOf());

        REQUIRE_TRUE(Wq->rankOf() == Wk->rankOf() && Wk->rankOf() == Wv->rankOf(), 0,
                     "multi_head_dot_product_attention: Input projections weights must have the same rank. "
                     "But got Wq = %s, Wk = %s, Wv = %s", ShapeUtils::shapeAsString(Wq).c_str(),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

        REQUIRE_TRUE(Wq->sizeAt(0) == Wk->sizeAt(0) && Wk->sizeAt(0) == Wv->sizeAt(0), 0,
                     "multi_head_dot_product_attention: Projections weights must have the same number of attention heads. "
                     "But got Wq = %s, Wk = %s, Wv = %s", ShapeUtils::shapeAsString(Wq).c_str(),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

        REQUIRE_TRUE(Wo->rankOf() == 2, 0,
                     "multi_head_dot_product_attention: Output projection weights must have rank 2. "
                     "But got Wo = %s", ShapeUtils::shapeAsString(Wo).c_str());

        REQUIRE_TRUE(Wq->sizeAt(2) == queries->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Query projection matrix Wq has incompatible size to queries matrix."
                     "Expected Wq[2] = queries[1] = %i, but got Wq = %s, queries = %s ", queries->sizeAt(1),
                     ShapeUtils::shapeAsString(Wq).c_str(), ShapeUtils::shapeAsString(queries).c_str());

        REQUIRE_TRUE(Wk->sizeAt(2) == keys->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Key projection matrix Wk has incompatible size to keys matrix."
                     "Expected Wk[2] = keys[1] = %i, but got Wk = %s, keys = %s ", keys->sizeAt(1),
                     ShapeUtils::shapeAsString(Wk).c_str(), ShapeUtils::shapeAsString(keys).c_str());

        REQUIRE_TRUE(Wv->sizeAt(2) == values->sizeAt(1), 0,
                     "multi_head_dot_product_attention: Value projection matrix Wv has incompatible size to values matrix."
                     "Expected Wv[2] = values[1] = %i, but got Wv = %s, values = %s ", values->sizeAt(1),
                     ShapeUtils::shapeAsString(Wv).c_str(), ShapeUtils::shapeAsString(values).c_str());

        REQUIRE_TRUE(Wo->sizeAt(0) == (Wv->sizeAt(1) * Wv->sizeAt(0)), 0,
                     "multi_head_dot_product_attention: Output projection matrix Wo has incompatible size to attention result."
                     "Expected Wo[0] = Wv[0] * Wv[1] = %i, but got Wo = %s, Wv = %", (Wv->sizeAt(1) * Wv->sizeAt(0)),
                     ShapeUtils::shapeAsString(Wo).c_str(), ShapeUtils::shapeAsString(Wv).c_str());

        nd4j::ops::matmul mmul;
        // Project queries, keys, values
        auto projectedQueries = AttentionHelper::multiHeadProject(queries, Wq);
        auto projectedKeys = AttentionHelper::multiHeadProject(keys, Wk);
        auto projectedValues = AttentionHelper::multiHeadProject(values, Wv);

        // Apply Attention
        nd4j::ops::dot_product_attention attention;
        auto results = attention.execute({projectedQueries, projectedKeys, projectedValues, mask}, {}, {normalization, 0}, {});


        // Project attention results
        auto attnResults = results->at(0);
        auto attnResultsPerm = attnResults->permute({0, 3, 1, 2});
        auto reshaped = attnResultsPerm->reshape(attnResults->ordering(), {miniBatchSize * queryCount, numHeads * projectedValuesSize});

        // dLdWo
        auto epsPerm = eps->permute({0, 2, 1});
        auto epsPostReshape = epsPerm->reshape(eps->ordering(), {miniBatchSize * queryCount, outSize});
        nd4j::ops::matmul_bp matmulBp;
        auto matMulBpRes = matmulBp.execute({reshaped, Wo, epsPostReshape}, {}, {}, {});
        auto dLdPreWo = matMulBpRes->at(0);
        dLdWo->assign(matMulBpRes->at(1));

        // dLdAttn
        dLdPreWo->reshapei({miniBatchSize, queryCount, numHeads, projectedValues->sizeAt(2)});
        dLdPreWo->permutei({0, 2, 3, 1});

        nd4j::ops::dot_product_attention_bp attentionBp;
        auto attnBp = attentionBp.execute({projectedQueries, projectedKeys, projectedValues, dLdPreWo, mask}, {}, {normalization}, {});

        AttentionHelper::multiHeadProjectBp(queries, Wq, attnBp->at(0), dLdq, dLdWq);
        AttentionHelper::multiHeadProjectBp(keys, Wk, attnBp->at(1), dLdk, dLdWk);
        AttentionHelper::multiHeadProjectBp(values, Wv, attnBp->at(2), dLdv, dLdWv);

        delete projectedQueries;
        delete projectedKeys;
        delete projectedValues;
        delete results;
        delete attnResultsPerm;
        delete reshaped;
        delete epsPerm;
        delete epsPostReshape;
        delete attnBp;
        delete matMulBpRes;

        return Status::OK();
    }

    DECLARE_TYPES(multi_head_dot_product_attention_bp) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(multi_head_dot_product_attention_bp) {
        Nd4jLong *dLdq_shape;
        COPY_SHAPE(inputShape->at(0), dLdq_shape);
        Nd4jLong *dLdk_shape;
        COPY_SHAPE(inputShape->at(1), dLdk_shape);
        Nd4jLong *dLdv_shape;
        COPY_SHAPE(inputShape->at(2), dLdv_shape);
        Nd4jLong *dLdWq_shape;
        COPY_SHAPE(inputShape->at(3), dLdWq_shape);
        Nd4jLong *dLdWk_shape;
        COPY_SHAPE(inputShape->at(4), dLdWk_shape);
        Nd4jLong *dLdWv_shape;
        COPY_SHAPE(inputShape->at(5), dLdWv_shape);
        Nd4jLong *dLdWo_shape;
        COPY_SHAPE(inputShape->at(6), dLdWo_shape);

        return SHAPELIST(dLdq_shape, dLdk_shape, dLdv_shape, dLdWq_shape, dLdWk_shape, dLdWv_shape, dLdWo_shape);
    }

}
}

#endif