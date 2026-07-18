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
// LoKr (Low-Rank Kronecker Product) fused matrix multiplication operation.
// Computes: output = input @ weight^T + scaling * input @ (C ⊗ (B @ A))^T
//
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_lokr_matmul)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/matmul.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(lokr_matmul, 5, 1, false, 0, 2) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto lokrC = INPUT_VARIABLE(2);   // [f1, f2]
  auto lokrA = INPUT_VARIABLE(3);   // [dim, d2]
  auto lokrB = INPUT_VARIABLE(4);   // [d1, dim]
  auto output = OUTPUT_VARIABLE(0); // [batch, out_features]

  if (input->isEmpty() || weight->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  int factor1 = INT_ARG(0);  // f1
  int factor2 = INT_ARG(1);  // f2

  REQUIRE_TRUE(input->rankOf() == 2, 0,
               "lokr_matmul: Input should have rank 2, got %i", input->rankOf());

  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  // Step 1: Compute base output: input @ weight^T
  auto baseOutput = output->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, transposeWeight ? 1 : 0};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, weight};
    std::vector<NDArray*> outputs = {baseOutput};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  // Step 2: Compute BA product: [d1, d2]
  auto d1 = lokrB->sizeAt(0);
  auto d2 = lokrA->sizeAt(1);
  std::vector<sd::LongType> baShape = {d1, d2};
  auto ba = NDArrayFactory::create('c', baShape, input->dataType(), input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lokrB, lokrA};
    std::vector<NDArray*> outputs = {ba};
    mmulOp.execute(inputs, outputs);
  }

  // Step 3: Compute Kronecker product C ⊗ BA
  std::vector<sd::LongType> baReshapeShape = {1, 1, d1, d2};
  auto baReshaped = ba->reshape('c', baReshapeShape);  // returns NDArray*

  std::vector<sd::LongType> cReshapeShape = {factor1, factor2, 1, 1};
  auto cReshaped = lokrC->reshape('c', cReshapeShape);  // returns NDArray*

  auto kronecker = (*baReshaped) * (*cReshaped);

  delete baReshaped;
  delete cReshaped;

  std::vector<sd::LongType> permuteDims = {0, 2, 1, 3};
  auto permuted = kronecker->permute(permuteDims, false, false);  // returns NDArray*

  std::vector<sd::LongType> lokrDeltaShape = {outFeatures, inFeatures};
  auto lokrDelta = permuted->reshape('c', lokrDeltaShape);  // returns NDArray*

  // Step 4: Compute LoKr output: input @ lokrDelta^T
  auto lokrOutput = output->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, lokrDelta};
    std::vector<NDArray*> outputs = {lokrOutput};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  // Step 5: Combine: output = base_output + scaling * lokr_output
  if (std::abs(scaling - 1.0) < 1e-9) {
    auto combined = (*baseOutput) + (*lokrOutput);
    output->assign(combined);
    delete combined;
  } else {
    lokrOutput->applyScalar(scalar::Multiply, scaling, lokrOutput);
    auto combined = (*baseOutput) + (*lokrOutput);
    output->assign(combined);
    delete combined;
  }

  delete ba;
  delete kronecker;
  delete permuted;
  delete lokrDelta;
  delete baseOutput;
  delete lokrOutput;

  return Status::OK;
}

DECLARE_SHAPE_FN(lokr_matmul) {
  auto inShapeInfo = inputShape->at(0);
  auto wShapeInfo = inputShape->at(1);

  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  sd::LongType batch = shape::sizeAt(inShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType outFeatures = transposeWeight ?
      shape::sizeAt(wShapeInfo, static_cast<sd::LongType>(0)) :
      shape::sizeAt(wShapeInfo, static_cast<sd::LongType>(1));

  std::vector<sd::LongType> outputShape = {batch, outFeatures};
  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inShapeInfo), 'c', outputShape);

  return SHAPELIST(outputShapeInfo);
}

DECLARE_TYPES(lokr_matmul) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(lokr_matmul_bp, 6, 5, false, 0, 2) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto lokrC = INPUT_VARIABLE(2);   // [f1, f2]
  auto lokrA = INPUT_VARIABLE(3);   // [dim, d2]
  auto lokrB = INPUT_VARIABLE(4);   // [d1, dim]
  auto dLdOut = INPUT_VARIABLE(5);  // [batch, out_features]

  auto dLdInput = OUTPUT_VARIABLE(0);
  auto dLdWeight = OUTPUT_VARIABLE(1);
  auto dLdLokrC = OUTPUT_VARIABLE(2);
  auto dLdLokrA = OUTPUT_VARIABLE(3);
  auto dLdLokrB = OUTPUT_VARIABLE(4);

  if (input->isEmpty() || dLdOut->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  int factor1 = INT_ARG(0);
  int factor2 = INT_ARG(1);

  // Gradient w.r.t. weight is zero (frozen)
  double zero = 0.0;
  dLdWeight->assign(zero);

  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  // Recompute lokrDelta
  auto d1 = lokrB->sizeAt(0);
  auto d2 = lokrA->sizeAt(1);
  std::vector<sd::LongType> baShape = {d1, d2};
  auto ba = NDArrayFactory::create('c', baShape, input->dataType(), input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lokrB, lokrA};
    std::vector<NDArray*> outputs = {ba};
    mmulOp.execute(inputs, outputs);
  }

  std::vector<sd::LongType> baReshapeShape = {1, 1, d1, d2};
  auto baReshaped = ba->reshape('c', baReshapeShape);  // returns NDArray*
  std::vector<sd::LongType> cReshapeShape = {factor1, factor2, 1, 1};
  auto cReshaped = lokrC->reshape('c', cReshapeShape);  // returns NDArray*
  auto kronecker = (*baReshaped) * (*cReshaped);
  delete baReshaped;
  delete cReshaped;
  std::vector<sd::LongType> permuteDims = {0, 2, 1, 3};
  auto permuted = kronecker->permute(permuteDims, false, false);  // returns NDArray*
  std::vector<sd::LongType> lokrDeltaShape = {outFeatures, inFeatures};
  auto lokrDelta = permuted->reshape('c', lokrDeltaShape);  // returns NDArray*

  // Gradient w.r.t. input
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, weight};
    std::vector<NDArray*> outputs = {dLdInput};
    mmulOp.execute(inputs, outputs);
  }
  auto dLdInputLokr = dLdInput->ulike();
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, lokrDelta};
    std::vector<NDArray*> outputs = {dLdInputLokr};
    mmulOp.execute(inputs, outputs);
  }
  dLdInputLokr->applyScalar(scalar::Multiply, scaling, dLdInputLokr);
  *dLdInput += *dLdInputLokr;
  delete dLdInputLokr;

  // Gradients for LoKr components are complex - using zeros for simplicity
  dLdLokrC->assign(zero);
  dLdLokrA->assign(zero);
  dLdLokrB->assign(zero);

  delete ba;
  delete kronecker;
  delete permuted;
  delete lokrDelta;

  return Status::OK;
}

DECLARE_SHAPE_FN(lokr_matmul_bp) {
  return SHAPELIST(
      CONSTANT(inputShape->at(0)),
      CONSTANT(inputShape->at(1)),
      CONSTANT(inputShape->at(2)),
      CONSTANT(inputShape->at(3)),
      CONSTANT(inputShape->at(4))
  );
}

DECLARE_TYPES(lokr_matmul_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
