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
// LoRA (Low-Rank Adaptation) fused matrix multiplication operation.
// Computes: output = input @ weight^T + scaling * (input @ A^T @ B^T)
//
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_lora_matmul)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/matmul.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(lora_matmul, 4, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features] OR [B, S, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto loraA = INPUT_VARIABLE(2);   // [r, in_features]
  auto loraB = INPUT_VARIABLE(3);   // [out_features, r]
  auto output = OUTPUT_VARIABLE(0); // [batch, out_features] OR [B, S, out_features]

  if (input->isEmpty() || weight->isEmpty() || loraA->isEmpty() || loraB->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  int inputRank = input->rankOf();
  REQUIRE_TRUE(inputRank == 2 || inputRank == 3, 0,
               "lora_matmul: Input array should have rank 2 or 3, got %i", inputRank);

  // For rank-3 [B,S,in_features], flatten to [B*S, in_features] for matmuls
  bool rank3 = (inputRank == 3);
  sd::LongType B = 1, S = 1;
  if (rank3) {
    B = input->sizeAt(0);
    S = input->sizeAt(1);
  }
  sd::LongType inFeatures = input->sizeAt(inputRank - 1);
  sd::LongType outFeatures = transposeWeight ? weight->sizeAt(0) : weight->sizeAt(1);
  sd::LongType M2d = rank3 ? (B * S) : input->sizeAt(0);

  NDArray* input2d   = nullptr;
  NDArray* output2d  = nullptr;
  bool reshapedInput = false;

  if (rank3) {
    std::vector<sd::LongType> s2 = {M2d, inFeatures};
    input2d = input->reshape('c', s2);
    reshapedInput = true;
    std::vector<sd::LongType> o2 = {M2d, outFeatures};
    output2d = NDArrayFactory::create('c', o2, input->dataType(), input->getContext());
  } else {
    input2d  = input;
    output2d = output;  // write directly for rank-2 (like original)
  }

  // Step 1: Compute base output: input2d @ weight^T  (or weight if !transposeWeight)
  auto baseOutput2d = output2d->ulike();  // returns NDArray*
  if (transposeWeight) {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input2d, weight};
    std::vector<NDArray*> outputs = {baseOutput2d};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  } else {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {input2d, weight};
    std::vector<NDArray*> outputs = {baseOutput2d};
    mmulOp.execute(inputs, outputs);
  }

  // Step 2: Compute LoRA contribution  [M2d, r] then [M2d, out_features]
  auto r = loraA->sizeAt(0);

  std::vector<sd::LongType> temp1Shape = {M2d, r};
  auto temp1 = NDArrayFactory::create('c', temp1Shape, input->dataType(), input->getContext());

  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input2d, loraA};
    std::vector<NDArray*> outputs = {temp1};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  auto loraOutput2d = output2d->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {temp1, loraB};
    std::vector<NDArray*> outputs = {loraOutput2d};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  // Step 3: Combine into output2d
  if (std::abs(scaling - 1.0) < 1e-9) {
    auto combined = (*baseOutput2d) + (*loraOutput2d);
    output2d->assign(combined);
    delete combined;
  } else {
    loraOutput2d->applyScalar(scalar::Multiply, scaling, loraOutput2d);
    auto combined = (*baseOutput2d) + (*loraOutput2d);
    output2d->assign(combined);
    delete combined;
  }

  delete temp1;
  delete baseOutput2d;
  delete loraOutput2d;

  // Step 4: If rank-3, reshape output2d [M2d,out_features] → [B,S,out_features]
  if (rank3) {
    std::vector<sd::LongType> outShape3 = {B, S, outFeatures};
    auto output3d = output2d->reshape('c', outShape3);
    output->assign(output3d);
    delete output3d;
    delete output2d;
    delete input2d;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(lora_matmul) {
  auto inShapeInfo = inputShape->at(0);
  auto wShapeInfo  = inputShape->at(1);
  int  inRank      = shape::rank(inShapeInfo);

  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  sd::LongType outFeatures = transposeWeight ?
      shape::sizeAt(wShapeInfo, static_cast<sd::LongType>(0)) :
      shape::sizeAt(wShapeInfo, static_cast<sd::LongType>(1));

  std::vector<sd::LongType> outputShape;
  if (inRank == 3) {
    // [B, S, in_features] → [B, S, out_features]
    outputShape = {shape::sizeAt(inShapeInfo, 0),
                   shape::sizeAt(inShapeInfo, 1),
                   outFeatures};
  } else {
    // [batch, in_features] → [batch, out_features]
    outputShape = {shape::sizeAt(inShapeInfo, 0), outFeatures};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inShapeInfo), 'c', outputShape);

  return SHAPELIST(outputShapeInfo);
}

DECLARE_TYPES(lora_matmul) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(lora_matmul_bp, 5, 4, false, 0, 0) {
  auto input  = INPUT_VARIABLE(0);  // [batch, in_features] or [B, S, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto loraA  = INPUT_VARIABLE(2);  // [r, in_features]
  auto loraB  = INPUT_VARIABLE(3);  // [out_features, r]
  auto dLdOut = INPUT_VARIABLE(4);  // [batch, out_features] or [B, S, out_features]

  auto dLdInput  = OUTPUT_VARIABLE(0);
  auto dLdWeight = OUTPUT_VARIABLE(1);
  auto dLdLoraA  = OUTPUT_VARIABLE(2);
  auto dLdLoraB  = OUTPUT_VARIABLE(3);

  if (input->isEmpty() || dLdOut->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  // Gradient w.r.t. weight is zero (frozen in LoRA training)
  double zeroD = 0.0;
  dLdWeight->assign(zeroD);

  int inputRank = input->rankOf();
  bool rank3 = (inputRank == 3);
  sd::LongType B = 1, S = 1;
  if (rank3) {
    B = input->sizeAt(0);
    S = input->sizeAt(1);
  }
  sd::LongType inFeatures  = input->sizeAt(inputRank - 1);
  sd::LongType outFeatures = transposeWeight ? weight->sizeAt(0) : weight->sizeAt(1);
  sd::LongType M2d = rank3 ? (B * S) : input->sizeAt(0);
  auto r = loraA->sizeAt(0);

  // Flatten to 2D for matmul computations
  NDArray* input2d  = nullptr;
  NDArray* dLdOut2d = nullptr;
  NDArray* dLdInput2d = nullptr;
  bool reshapedInput = false;

  if (rank3) {
    std::vector<sd::LongType> si = {M2d, inFeatures};
    input2d  = input->reshape('c', si);
    std::vector<sd::LongType> sg = {M2d, outFeatures};
    dLdOut2d = dLdOut->reshape('c', sg);
    std::vector<sd::LongType> so = {M2d, inFeatures};
    dLdInput2d = NDArrayFactory::create('c', so, input->dataType(), input->getContext());
    reshapedInput = true;
  } else {
    input2d    = input;
    dLdOut2d   = dLdOut;
    dLdInput2d = dLdInput;
  }

  // ── Gradient w.r.t. input from base path ──────────────────────────────────
  if (transposeWeight) {
    // weight is [out_features, in_features]; dLdOut2d[M2d,out] @ weight[out,in] → [M2d,in]
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {dLdOut2d, weight};
    std::vector<NDArray*> outputs = {dLdInput2d};
    mmulOp.execute(inputs, outputs);
  } else {
    // weight is [in_features, out_features]; need its transpose
    auto weightT = weight->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {dLdOut2d, weightT};
    std::vector<NDArray*> outputs = {dLdInput2d};
    mmulOp.execute(inputs, outputs);
    delete weightT;
  }

  // ── LoRA path ──────────────────────────────────────────────────────────────
  // temp[M2d, r] = dLdOut2d[M2d, out] @ loraB[out, r]
  auto temp = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, r},
                                     input->dataType(), input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {dLdOut2d, loraB};
    std::vector<NDArray*> outputs = {temp};
    mmulOp.execute(inputs, outputs);
  }

  // dInput_lora[M2d, in] = temp[M2d, r] @ loraA[r, in]
  auto dLdInputLora2d = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, inFeatures},
                                                input->dataType(), input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {temp, loraA};
    std::vector<NDArray*> outputs = {dLdInputLora2d};
    mmulOp.execute(inputs, outputs);
  }
  dLdInputLora2d->applyScalar(scalar::Multiply, scaling, dLdInputLora2d);
  *dLdInput2d += *dLdInputLora2d;
  delete dLdInputLora2d;

  // ── Gradient w.r.t. loraA: [r, in] = tempᵀ[r,M2d] @ input2d[M2d,in] ─────
  {
    auto tempT = temp->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {tempT, input2d};
    std::vector<NDArray*> outputs = {dLdLoraA};
    mmulOp.execute(inputs, outputs);
    delete tempT;
  }
  dLdLoraA->applyScalar(scalar::Multiply, scaling, dLdLoraA);

  // ── Gradient w.r.t. loraB: [out, r] = dLdOut2dᵀ[out,M2d] @ temp1[M2d,r] ─
  // where temp1[M2d,r] = input2d[M2d,in] @ loraAᵀ[in,r]
  auto temp1 = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, r},
                                      input->dataType(), input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};  // transposeB=true for loraA
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs  = {input2d, loraA};
    std::vector<NDArray*> outputs = {temp1};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }
  {
    auto dLdOut2dT = dLdOut2d->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs  = {dLdOut2dT, temp1};
    std::vector<NDArray*> outputs = {dLdLoraB};
    mmulOp.execute(inputs, outputs);
    delete dLdOut2dT;
  }
  dLdLoraB->applyScalar(scalar::Multiply, scaling, dLdLoraB);

  delete temp;
  delete temp1;

  // ── Reshape dLdInput back to rank-3 if needed ─────────────────────────────
  if (rank3) {
    std::vector<sd::LongType> inShape3 = {B, S, inFeatures};
    auto dLdInput3d = dLdInput2d->reshape('c', inShape3);
    dLdInput->assign(dLdInput3d);
    delete dLdInput3d;
    delete dLdInput2d;
    delete input2d;
    delete dLdOut2d;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(lora_matmul_bp) {
  return SHAPELIST(
      CONSTANT(inputShape->at(0)),
      CONSTANT(inputShape->at(1)),
      CONSTANT(inputShape->at(2)),
      CONSTANT(inputShape->at(3))
  );
}

DECLARE_TYPES(lora_matmul_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
