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
// DoRA (Weight-Decomposed Low-Rank Adaptation) fused matrix multiplication.
// Computes: output = input @ (m * (W + scaling * B @ A) / ||W + scaling * B @ A||)^T
//
// DoRA decomposes weight updates into magnitude and direction components.
//
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dora_matmul)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/matmul.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dora_matmul, 5, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);      // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);     // [out_features, in_features]
  auto loraA = INPUT_VARIABLE(2);      // [r, in_features]
  auto loraB = INPUT_VARIABLE(3);      // [out_features, r]
  auto magnitude = INPUT_VARIABLE(4);  // [out_features] or [out_features, 1]
  auto output = OUTPUT_VARIABLE(0);    // [batch, out_features]

  if (input->isEmpty() || weight->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  double eps = block.getTArguments()->size() > 2 ? T_ARG(2) : 1e-8;

  REQUIRE_TRUE(input->rankOf() == 2, 0,
               "dora_matmul: Input should have rank 2, got %i", input->rankOf());

  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  // Step 1: Compute LoRA delta: scaling * B @ A
  auto loraDelta = weight->ulike();  // [out_features, in_features]
  {
    sd::ops::matmul mmulOp;
    auto status = mmulOp.execute({loraB, loraA}, {loraDelta});
    if (status != Status::OK) {
      delete loraDelta;
      return status;
    }
  }
  loraDelta->applyScalar(scalar::Multiply, scaling, loraDelta);

  // Step 2: Compute effective weight: W + scaling * B @ A
  auto wEff = (*weight) + (*loraDelta);

  // Step 3: Compute column-wise L2 norm
  auto wEffSquared = (*wEff) * (*wEff);
  std::vector<sd::LongType> sumDims = {1};
  auto normSquared = wEffSquared->reduceAlongDimension(reduce::Sum, &sumDims, true);  // returns NDArray*
  normSquared->applyScalar(scalar::Add, eps, normSquared);
  auto norm = normSquared->transform(transform::Sqrt);  // returns NDArray*

  // Step 4: Normalize: direction = W_eff / ||W_eff||
  auto direction = (*wEff) / (*norm);

  // Step 5: Apply magnitude
  NDArray* magExpanded = nullptr;
  bool deleteMag = false;
  if (magnitude->rankOf() == 1) {
    std::vector<sd::LongType> magShape = {outFeatures, 1};
    magExpanded = magnitude->reshape('c', magShape);
    deleteMag = true;
  } else {
    magExpanded = magnitude;
  }

  // Final weight: m * direction
  auto finalWeight = (*direction) * (*magExpanded);

  // Step 6: Compute output: input @ finalWeight^T
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};  // transA=0, transB=1
    auto status = mmulOp.execute({input, finalWeight}, {output}, tArgs, iArgs);
    if (status != Status::OK) {
      delete loraDelta;
      delete wEff;
      delete wEffSquared;
      delete normSquared;
      delete norm;
      delete direction;
      delete finalWeight;
      if (deleteMag) delete magExpanded;
      return status;
    }
  }

  // Cleanup
  delete loraDelta;
  delete wEff;
  delete wEffSquared;
  delete normSquared;
  delete norm;
  delete direction;
  delete finalWeight;
  if (deleteMag) delete magExpanded;

  return Status::OK;
}

DECLARE_SHAPE_FN(dora_matmul) {
  auto inShapeInfo = inputShape->at(0);
  auto wShapeInfo = inputShape->at(1);

  sd::LongType batch = shape::sizeAt(inShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType outFeatures = shape::sizeAt(wShapeInfo, static_cast<sd::LongType>(0));

  std::vector<sd::LongType> outShape = {batch, outFeatures};
  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inShapeInfo), 'c', outShape);

  return SHAPELIST(outputShapeInfo);
}

DECLARE_TYPES(dora_matmul) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(dora_matmul_bp, 6, 5, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);      // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);     // [out_features, in_features]
  auto loraA = INPUT_VARIABLE(2);      // [r, in_features]
  auto loraB = INPUT_VARIABLE(3);      // [out_features, r]
  auto magnitude = INPUT_VARIABLE(4);  // [out_features]
  auto dLdOut = INPUT_VARIABLE(5);     // [batch, out_features]

  auto dLdInput = OUTPUT_VARIABLE(0);     // [batch, in_features]
  auto dLdWeight = OUTPUT_VARIABLE(1);    // [out_features, in_features]
  auto dLdLoraA = OUTPUT_VARIABLE(2);     // [r, in_features]
  auto dLdLoraB = OUTPUT_VARIABLE(3);     // [out_features, r]
  auto dLdMagnitude = OUTPUT_VARIABLE(4); // [out_features]

  if (input->isEmpty() || dLdOut->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  double eps = block.getTArguments()->size() > 2 ? T_ARG(2) : 1e-8;

  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  // Gradient w.r.t. weight is zero (frozen)
  double zero = 0.0;
  dLdWeight->assign(zero);

  // Recompute forward pass intermediates
  auto loraDelta = weight->ulike();
  {
    sd::ops::matmul mmulOp;
    mmulOp.execute({loraB, loraA}, {loraDelta});
  }
  loraDelta->applyScalar(scalar::Multiply, scaling, loraDelta);

  auto wEff = (*weight) + (*loraDelta);
  auto wEffSquared = (*wEff) * (*wEff);
  std::vector<sd::LongType> sumDims = {1};
  auto normSquared = wEffSquared->reduceAlongDimension(reduce::Sum, &sumDims, true);  // returns NDArray*
  normSquared->applyScalar(scalar::Add, eps, normSquared);
  auto norm = normSquared->transform(transform::Sqrt);  // returns NDArray*
  auto direction = (*wEff) / (*norm);

  NDArray* magExpanded = nullptr;
  bool deleteMag = false;
  if (magnitude->rankOf() == 1) {
    std::vector<sd::LongType> magShape = {outFeatures, 1};
    magExpanded = magnitude->reshape('c', magShape);
    deleteMag = true;
  } else {
    magExpanded = magnitude;
  }

  auto finalWeight = (*direction) * (*magExpanded);

  // Gradient w.r.t. input
  {
    sd::ops::matmul mmulOp;
    mmulOp.execute({dLdOut, finalWeight}, {dLdInput});
  }

  // Gradient w.r.t. magnitude
  auto dLdOutT = dLdOut->transpose();
  auto dLdOutTimesInput = weight->ulike();
  {
    sd::ops::matmul mmulOp;
    mmulOp.execute({dLdOutT, input}, {dLdOutTimesInput});
  }

  auto gradMagTemp = (*dLdOutTimesInput) * (*direction);
  auto gradMag = gradMagTemp->reduceAlongDimension(reduce::Sum, &sumDims, false);  // returns NDArray*
  dLdMagnitude->assign(gradMag);

  // Gradient w.r.t. LoRA components (simplified)
  auto gradDirectionTemp = (*dLdOutTimesInput) * (*magExpanded);
  auto gradDirection = (*gradDirectionTemp) / (*norm);

  // dLdA = scaling * B^T @ gradDirection
  {
    auto loraBT = loraB->transpose();
    sd::ops::matmul mmulOp;
    mmulOp.execute({loraBT, gradDirection}, {dLdLoraA});
    delete loraBT;
  }
  dLdLoraA->applyScalar(scalar::Multiply, scaling, dLdLoraA);

  // dLdB = scaling * gradDirection @ A^T
  {
    auto loraAT = loraA->transpose();
    sd::ops::matmul mmulOp;
    mmulOp.execute({gradDirection, loraAT}, {dLdLoraB});
    delete loraAT;
  }
  dLdLoraB->applyScalar(scalar::Multiply, scaling, dLdLoraB);

  // Cleanup
  delete loraDelta;
  delete wEff;
  delete wEffSquared;
  delete normSquared;
  delete norm;
  delete direction;
  delete finalWeight;
  delete dLdOutT;
  delete dLdOutTimesInput;
  delete gradMagTemp;
  delete gradMag;
  delete gradDirectionTemp;
  delete gradDirection;
  if (deleteMag) delete magExpanded;

  return Status::OK;
}

DECLARE_SHAPE_FN(dora_matmul_bp) {
  return SHAPELIST(
      CONSTANT(inputShape->at(0)),
      CONSTANT(inputShape->at(1)),
      CONSTANT(inputShape->at(2)),
      CONSTANT(inputShape->at(3)),
      CONSTANT(inputShape->at(4))
  );
}

DECLARE_TYPES(dora_matmul_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
