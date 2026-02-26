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
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto loraA = INPUT_VARIABLE(2);   // [r, in_features]
  auto loraB = INPUT_VARIABLE(3);   // [out_features, r]
  auto output = OUTPUT_VARIABLE(0); // [batch, out_features]

  if (input->isEmpty() || weight->isEmpty() || loraA->isEmpty() || loraB->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  REQUIRE_TRUE(input->rankOf() == 2, 0,
               "lora_matmul: Input array should have rank 2, got %i", input->rankOf());

  // Step 1: Compute base output: input @ weight^T
  auto baseOutput = output->ulike();  // returns NDArray*
  if (transposeWeight) {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, weight};
    std::vector<NDArray*> outputs = {baseOutput};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  } else {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {input, weight};
    std::vector<NDArray*> outputs = {baseOutput};
    mmulOp.execute(inputs, outputs);
  }

  // Step 2: Compute LoRA contribution
  auto r = loraA->sizeAt(0);
  auto batch = input->sizeAt(0);

  std::vector<sd::LongType> temp1Shape = {batch, r};
  auto temp1 = NDArrayFactory::create<float>('c', temp1Shape, input->getContext());

  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, loraA};
    std::vector<NDArray*> outputs = {temp1};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  auto loraOutput = output->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {temp1, loraB};
    std::vector<NDArray*> outputs = {loraOutput};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  // Step 3: Combine
  if (std::abs(scaling - 1.0) < 1e-9) {
    auto combined = (*baseOutput) + (*loraOutput);
    output->assign(combined);
    delete combined;
  } else {
    loraOutput->applyScalar(scalar::Multiply, scaling, loraOutput);
    auto combined = (*baseOutput) + (*loraOutput);
    output->assign(combined);
    delete combined;
  }

  delete temp1;
  delete baseOutput;
  delete loraOutput;

  return Status::OK;
}

DECLARE_SHAPE_FN(lora_matmul) {
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

DECLARE_TYPES(lora_matmul) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(lora_matmul_bp, 5, 4, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto loraA = INPUT_VARIABLE(2);   // [r, in_features]
  auto loraB = INPUT_VARIABLE(3);   // [out_features, r]
  auto dLdOut = INPUT_VARIABLE(4);  // [batch, out_features]

  auto dLdInput = OUTPUT_VARIABLE(0);
  auto dLdWeight = OUTPUT_VARIABLE(1);
  auto dLdLoraA = OUTPUT_VARIABLE(2);
  auto dLdLoraB = OUTPUT_VARIABLE(3);

  if (input->isEmpty() || dLdOut->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  // Gradient w.r.t. weight is zero (frozen)
  double zero = 0.0;
  dLdWeight->assign(zero);

  // Gradient w.r.t. input from base path
  if (transposeWeight) {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, weight};
    std::vector<NDArray*> outputs = {dLdInput};
    mmulOp.execute(inputs, outputs);
  } else {
    auto weightT = weight->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, weightT};
    std::vector<NDArray*> outputs = {dLdInput};
    mmulOp.execute(inputs, outputs);
    delete weightT;
  }

  auto r = loraA->sizeAt(0);
  auto batch = input->sizeAt(0);

  // dLdOut @ B -> [batch, r]
  std::vector<sd::LongType> tempShape = {batch, r};
  auto temp = NDArrayFactory::create<float>('c', tempShape, input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, loraB};
    std::vector<NDArray*> outputs = {temp};
    mmulOp.execute(inputs, outputs);
  }

  // temp @ A -> [batch, in_features]
  auto dLdInputLora = dLdInput->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {temp, loraA};
    std::vector<NDArray*> outputs = {dLdInputLora};
    mmulOp.execute(inputs, outputs);
  }

  dLdInputLora->applyScalar(scalar::Multiply, scaling, dLdInputLora);
  *dLdInput += *dLdInputLora;
  delete dLdInputLora;

  // Gradient w.r.t. loraA
  {
    auto tempT = temp->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {tempT, input};
    std::vector<NDArray*> outputs = {dLdLoraA};
    mmulOp.execute(inputs, outputs);
    delete tempT;
  }
  dLdLoraA->applyScalar(scalar::Multiply, scaling, dLdLoraA);

  // Gradient w.r.t. loraB
  std::vector<sd::LongType> temp1Shape = {batch, r};
  auto temp1 = NDArrayFactory::create<float>('c', temp1Shape, input->getContext());
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, loraA};
    std::vector<NDArray*> outputs = {temp1};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  {
    auto dLdOutT = dLdOut->transpose();
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOutT, temp1};
    std::vector<NDArray*> outputs = {dLdLoraB};
    mmulOp.execute(inputs, outputs);
    delete dLdOutT;
  }
  dLdLoraB->applyScalar(scalar::Multiply, scaling, dLdLoraB);

  delete temp;
  delete temp1;

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
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
