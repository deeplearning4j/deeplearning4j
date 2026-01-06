/*
*  ******************************************************************************
*  *
*  *
*  * This program and the accompanying materials are made available under the
*  * terms of the Apache License, Version 2.0 which is available at
*  * https://www.apache.org/licenses/LICENSE-2.0.
*  *
*  * See the NOTICE file distributed with this work for additional
*  * information regarding copyright ownership.
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*  * License for the specific language governing permissions and limitations
*  * under the License.
*  *
*  * SPDX-License-Identifier: Apache-2.0
*  *****************************************************************************
*/

//
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dot_product_attention)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dot_product_attention, 3, -1, false, 0, 2) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

  auto output = OUTPUT_VARIABLE(0);
  NDArray *weights;
  bool outputWeights = INT_ARG(1);
  if (outputWeights) {
    weights = OUTPUT_VARIABLE(1);
  } else {
    auto weightShape = ShapeUtils::evalShapeForMatmul(keys->shapeInfo(), queries->shapeInfo(), true, false);
    weights = new NDArray('c', weightShape, values->dataType(), block.launchContext());
  }

  int normalization = INT_ARG(0);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
               "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
               "or rank 4 arrays for multi headed attention. But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
               "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
               "But got queries = %i, keys = %i, values = %i",
               queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

  REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
               "dot_product_attention: Queries and Keys must have the same feature size. "
               "But got queries = %i, keys = %i",
               queries->sizeAt(-2), keys->sizeAt(-2));

  REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
               "dot_product_attention: Keys and Values must have the same timestep length. "
               "But got keys = %i, values = %i",
               keys->sizeAt(-1), values->sizeAt(-1));

  sd::ops::matmul mmul;
  mmul.execute({keys, queries}, {weights}, {}, {1}, {});
  if (normalization) {
    *weights /= sqrt((double)keys->sizeAt(-2));
  }

  if (mask != nullptr && !mask->isEmpty()) {
    NDArray *reshapedMask;
    if (weights->rankOf() == 4) {
      std::vector<sd::LongType> shape = {mask->sizeAt(0), 1, mask->sizeAt(1), 1};
      reshapedMask = mask->reshape(mask->ordering(), shape);
    } else {
      std::vector<sd::LongType> shape = {mask->sizeAt(0), mask->sizeAt(1), 1};
      reshapedMask = mask->reshape(mask->ordering(), shape);
    }

    // The mask is 0 for positions we want to skip, and 1 for positions we want to keep.
    // We compute (1 - mask) * 1e9 to get 1e9 for skip positions and 0 for keep positions.
    // Subtracting this from weights pushes skip positions to large negative values,
    // which become ~0 after softmax. Keep positions are unchanged.
    auto* maskComplement = new NDArray(1.0 - *reshapedMask);
    *maskComplement *= 1e9;
    *weights -= *maskComplement;
    delete reshapedMask;
    delete maskComplement;
  }

  // Use explicit positive dimension for softmax (rank-2 for second-to-last dimension)
  int softmaxDim = weights->rankOf() - 2;
  sd::ops::softmax softmax;
  softmax.execute({weights}, std::vector<NDArray *>{weights}, {}, {softmaxDim}, {}, {}, true);

  mmul.execute({values, weights}, {output}, {}, {}, {});

  if (!outputWeights) {
    delete weights;
  }


  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention) {
  auto query_shape = inputShape->at(0);
  auto keys_shape = inputShape->at(1);
  auto values_shape = inputShape->at(2);

  auto weights_shape = ConstantShapeHelper::getInstance().createShapeInfo(
      sd::ArrayOptions::dataType(values_shape), 'c',
      ShapeUtils::evalShapeForMatmul(keys_shape, query_shape, true, false));
  auto output_shape = ConstantShapeHelper::getInstance().createShapeInfo(
      sd::ArrayOptions::dataType(values_shape), 'c',
      ShapeUtils::evalShapeForMatmul(values_shape, weights_shape, false, false));

  if (INT_ARG(1)) {
    return SHAPELIST(output_shape, weights_shape);
  } else {
    return SHAPELIST(output_shape);
  }
}

CUSTOM_OP_IMPL(dot_product_attention_bp, 4, 3, false, 0, 1) {
  auto queries = INPUT_VARIABLE(0);
  auto keys = INPUT_VARIABLE(1);
  auto values = INPUT_VARIABLE(2);
  auto eps = INPUT_VARIABLE(3);
  auto mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdk = OUTPUT_VARIABLE(1);
  auto dLdv = OUTPUT_VARIABLE(2);

  int normalization = INT_ARG(0);

  REQUIRE_TRUE(queries->rankOf() == keys->rankOf() && keys->rankOf() == values->rankOf(), 0,
               "dot_product_attention: Queries, Keys and Values must have same rank. "
               "But got queries = %s, keys = %s, values = %s",
               ShapeUtils::shapeAsString(queries).c_str(), ShapeUtils::shapeAsString(keys).c_str(),
               ShapeUtils::shapeAsString(values).c_str());

  REQUIRE_TRUE(queries->rankOf() == 3 || queries->rankOf() == 4, 0,
               "dot_product_attention: Queries, Keys and Values must be rank 3 arrays for single headed attention "
               "or rank 4 arrays for multi headed attention. But got rank = %i",
               queries->rankOf());

  REQUIRE_TRUE(queries->sizeAt(0) == keys->sizeAt(0) && keys->sizeAt(0) == values->sizeAt(0), 0,
               "dot_product_attention: Queries, Keys and Values must have the same mini batch size. "
               "But got queries = %i, keys = %i, values = %i",
               queries->sizeAt(0), keys->sizeAt(0), values->sizeAt(0));

  REQUIRE_TRUE(queries->sizeAt(-2) == keys->sizeAt(-2), 0,
               "dot_product_attention: Queries and Keys must have the same feature size. "
               "But got queries = %i, keys = %i",
               queries->sizeAt(-2), keys->sizeAt(-2));

  REQUIRE_TRUE(keys->sizeAt(-1) == values->sizeAt(-1), 0,
               "dot_product_attention: Keys and Values must have the same timestep length. "
               "But got keys = %i, values = %i",
               keys->sizeAt(-1), values->sizeAt(-1));

  double factor;
  if (normalization) factor = sqrt((double)keys->sizeAt(-2));

  auto weightShape = ShapeUtils::evalShapeForMatmul(keys->shapeInfo(), queries->shapeInfo(), true, false);

  sd::ops::matmul mmul;
  NDArray preSoftmax('c', weightShape, values->dataType(), block.launchContext());
  mmul.execute({keys, queries}, {&preSoftmax}, {}, {1}, {});

  if (normalization) preSoftmax /= factor;

  // Initialize reshapedMask to nullptr to avoid undefined behavior
  NDArray *reshapedMask = nullptr;
  if (mask != nullptr && !mask->isEmpty()) {
    if (preSoftmax.rankOf() == 4) {
      std::vector<sd::LongType> shape = {mask->sizeAt(0), 1, mask->sizeAt(1), 1};
      reshapedMask = mask->reshape(mask->ordering(), shape);
    } else {
      std::vector<sd::LongType> shape = {mask->sizeAt(0), mask->sizeAt(1), 1};
      reshapedMask = mask->reshape(mask->ordering(), shape);
    }

    // Apply mask: subtract large value for positions to skip (mask=0 means skip)
    // Note: The mask convention is: 0 = skip, 1 = keep
    // We subtract (1 - mask) * 1e9 to push skipped positions to -infinity
    auto* maskComplement = new NDArray(1.0 - *reshapedMask);
    *maskComplement *= 1e9;
    preSoftmax -= *maskComplement;
    delete maskComplement;
  }

  // Use explicit positive dimension for softmax (rank-2 for second-to-last dimension)
  // For rank 3 tensors: dim 1, for rank 4 tensors: dim 2
  int softmaxDim = preSoftmax.rankOf() - 2;

  NDArray weights('c', weightShape, values->dataType(), block.launchContext());
  sd::ops::softmax softmax;
  softmax.execute({&preSoftmax}, {&weights}, {}, {softmaxDim}, {});

  // Use heap allocation to avoid workspace issues
  NDArray dLdw('c', weightShape, values->dataType(), block.launchContext());
  sd::ops::matmul_bp mmul_bp;
  mmul_bp.execute({values, &weights, eps}, {dLdv, &dLdw}, {}, {}, {});

  NDArray dLds('c', weightShape, values->dataType(), block.launchContext());
  sd::ops::softmax_bp softmax_bp;
  softmax_bp.execute({&preSoftmax, &dLdw, &weights}, {&dLds}, {}, {softmaxDim}, {});

  if (normalization) dLds /= factor;


  // Note: No need to mask dLds - the softmax_bp already produces zero gradients
  // for masked positions because weights are zero there (softmax of -infinity = 0)

  // Compute gradients for keys and queries manually using matmul
  // Forward was: preSoftmax = matmul(keys, queries, {transX=1}) = K^T @ Q
  // Shapes: K=[batch,feat,Tk], Q=[batch,feat,Tq], preSoftmax=[batch,Tk,Tq]
  //
  // For Z = K^T @ Q where Z[b,i,j] = sum_f K[b,f,i] * Q[b,f,j]:
  //   dL/dK[b,f,i] = sum_j dL/dZ[b,i,j] * Q[b,f,j] = (Q @ dL/dZ^T)[b,f,i]
  //   dL/dQ[b,f,j] = sum_i dL/dZ[b,i,j] * K[b,f,i] = (K @ dL/dZ)[b,f,j]
  //
  // Using direct matmul instead of matmul_bp for clarity:
  //   dL/dK = Q @ dL/dZ^T = matmul(Q, dLds, {0, 1, 0}) where transY=1 gives dLds^T
  //   dL/dQ = K @ dL/dZ = matmul(K, dLds, {0, 0, 0})
  sd::ops::matmul mmul_dk;
  mmul_dk.execute({queries, &dLds}, {dLdk}, {}, {0, 1, 0}, {});  // Q @ dLds^T

  sd::ops::matmul mmul_dq;
  mmul_dq.execute({keys, &dLds}, {dLdq}, {}, {0, 0, 0}, {});  // K @ dLds

  // Only delete if it was allocated
  if (reshapedMask != nullptr) {
    delete reshapedMask;
  }

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)));
}

}  // namespace ops
}  // namespace sd

#endif
