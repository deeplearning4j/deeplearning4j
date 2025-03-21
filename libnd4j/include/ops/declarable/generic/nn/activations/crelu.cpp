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
#if NOT_EXCLUDED(OP_crelu)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/transforms.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(crelu, 1, 1, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);

  REQUIRE_TRUE(x->isR(), 0, "CRELU: input must be real type");

  auto tmp = x->dup(x->ordering());
  tmp.applyTransform(transform::Neg, &tmp);

  auto z = OUTPUT_VARIABLE(0);

  helpers::concat(block.launchContext(), {x, &tmp}, *z, x->rankOf() - 1);

  // TODO: make this configurable?
  double threshold = 0.0;
  z->applyScalar(scalar::RELU, threshold, z);

  STORE_RESULT(z);

  return Status::OK;
}

DECLARE_TYPES(crelu) { getOpDescriptor()->setAllowedInputTypes(0, ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(crelu) {
  auto inShape = inputShape->at(0);
  std::vector<LongType> shape;
  for (int e = 0; e < shape::rank(inShape); e++) shape.emplace_back(shape::shapeOf(inShape)[e]);

  shape[shape.size() - 1] *= 2;
  auto newShape =
      ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), shape);

  return SHAPELIST(newShape);
}

CUSTOM_OP_IMPL(crelu_bp, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto epsilonNext = INPUT_VARIABLE(1);
  auto epsilon = OUTPUT_VARIABLE(0);

  // at first step we build fwd activation
  crelu op;
  auto tmpResult = op.evaluate({input});
  if (tmpResult.status() != Status::OK) return tmpResult.status();

  auto actv = tmpResult.at(0);

  // now we do RELU backward pass
  helpers::reluDerivative(block.launchContext(), actv, epsilonNext);
  // now we split updated array into 2 chunks along last dimension
  concat_bp opc;
  auto dec = opc.evaluate({input, input, actv}, {-1});
  if (dec.status() != Status::OK) return dec.status();

  // and now we subtract two parts of epsilons and pass result out
  auto pos = dec.at(0);
  auto neg = dec.at(1);

  pos->applyPairwiseTransform(pairwise::Subtract, neg, epsilon);

  return Status::OK;
}

DECLARE_TYPES(crelu_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF});
}

DECLARE_SHAPE_FN(crelu_bp) {
  auto inShape = inputShape->at(0);
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
  return ret;
}
}  // namespace ops
}  // namespace sd

#endif
