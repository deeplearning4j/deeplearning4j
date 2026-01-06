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
// LoHa (Low-Rank Hadamard Product) fused matrix multiplication operation.
// Computes: output = input @ weight^T + scaling * input @ ((B1 @ A1) ⊙ (B2 @ A2))^T
//
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_loha_matmul)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/matmul.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(loha_matmul, 6, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto lohaA1 = INPUT_VARIABLE(2);  // [dim, in_features]
  auto lohaB1 = INPUT_VARIABLE(3);  // [out_features, dim]
  auto lohaA2 = INPUT_VARIABLE(4);  // [dim, in_features]
  auto lohaB2 = INPUT_VARIABLE(5);  // [out_features, dim]
  auto output = OUTPUT_VARIABLE(0); // [batch, out_features]

  if (input->isEmpty() || weight->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  REQUIRE_TRUE(input->rankOf() == 2, 0,
               "loha_matmul: Input should have rank 2, got %i", input->rankOf());

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

  // Step 2: Compute LoHa products
  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  std::vector<sd::LongType> prodShape = {outFeatures, inFeatures};
  auto prod1 = NDArrayFactory::create<float>('c', prodShape, input->getContext());
  auto prod2 = NDArrayFactory::create<float>('c', prodShape, input->getContext());

  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB1, lohaA1};
    std::vector<NDArray*> outputs = {prod1};
    mmulOp.execute(inputs, outputs);
  }
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB2, lohaA2};
    std::vector<NDArray*> outputs = {prod2};
    mmulOp.execute(inputs, outputs);
  }

  // Step 3: Hadamard product: lohaDelta = prod1 * prod2
  auto lohaDelta = new NDArray(*prod1 * *prod2);

  // Step 4: Compute LoHa output: input @ lohaDelta^T
  auto lohaOutput = output->ulike();  // returns NDArray*
  {
    sd::ops::matmul mmulOp;
    std::vector<double> tArgs;
    std::vector<sd::LongType> iArgs = {0, 1};
    std::vector<bool> bArgs;
    std::vector<NDArray*> inputs = {input, lohaDelta};
    std::vector<NDArray*> outputs = {lohaOutput};
    mmulOp.execute(inputs, outputs, tArgs, iArgs, bArgs);
  }

  // Step 5: Combine: output = base_output + scaling * loha_output
  if (std::abs(scaling - 1.0) < 1e-9) {
    output->assign(*baseOutput + *lohaOutput);
  } else {
    lohaOutput->applyScalar(scalar::Multiply, scaling, lohaOutput);
    output->assign(*baseOutput + *lohaOutput);
  }

  delete prod1;
  delete prod2;
  delete lohaDelta;
  delete baseOutput;
  delete lohaOutput;

  return Status::OK;
}

DECLARE_SHAPE_FN(loha_matmul) {
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

DECLARE_TYPES(loha_matmul) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(loha_matmul_bp, 7, 6, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);   // [batch, in_features]
  auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
  auto lohaA1 = INPUT_VARIABLE(2);  // [dim, in_features]
  auto lohaB1 = INPUT_VARIABLE(3);  // [out_features, dim]
  auto lohaA2 = INPUT_VARIABLE(4);  // [dim, in_features]
  auto lohaB2 = INPUT_VARIABLE(5);  // [out_features, dim]
  auto dLdOut = INPUT_VARIABLE(6);  // [batch, out_features]

  auto dLdInput = OUTPUT_VARIABLE(0);
  auto dLdWeight = OUTPUT_VARIABLE(1);
  auto dLdLohaA1 = OUTPUT_VARIABLE(2);
  auto dLdLohaB1 = OUTPUT_VARIABLE(3);
  auto dLdLohaA2 = OUTPUT_VARIABLE(4);
  auto dLdLohaB2 = OUTPUT_VARIABLE(5);

  if (input->isEmpty() || dLdOut->isEmpty()) {
    return Status::OK;
  }

  double scaling = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
  bool transposeWeight = block.getBArguments()->size() > 0 ? B_ARG(0) : true;

  // Gradient w.r.t. weight is zero (frozen)
  double zero = 0.0;
  dLdWeight->assign(zero);

  // Compute products
  auto outFeatures = weight->sizeAt(0);
  auto inFeatures = weight->sizeAt(1);

  std::vector<sd::LongType> prodShape = {outFeatures, inFeatures};
  auto prod1 = NDArrayFactory::create<float>('c', prodShape, input->getContext());
  auto prod2 = NDArrayFactory::create<float>('c', prodShape, input->getContext());

  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB1, lohaA1};
    std::vector<NDArray*> outputs = {prod1};
    mmulOp.execute(inputs, outputs);
  }
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB2, lohaA2};
    std::vector<NDArray*> outputs = {prod2};
    mmulOp.execute(inputs, outputs);
  }

  auto lohaDelta = new NDArray(*prod1 * *prod2);

  // Gradient w.r.t. input
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, weight};
    std::vector<NDArray*> outputs = {dLdInput};
    mmulOp.execute(inputs, outputs);
  }
  auto dLdInputLoha = dLdInput->ulike();
  {
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOut, lohaDelta};
    std::vector<NDArray*> outputs = {dLdInputLoha};
    mmulOp.execute(inputs, outputs);
  }
  dLdInputLoha->applyScalar(scalar::Multiply, scaling, dLdInputLoha);
  *dLdInput += *dLdInputLoha;
  delete dLdInputLoha;

  // Gradient w.r.t. LoHa components
  auto dLdOutTimesInput = NDArrayFactory::create<float>('c', prodShape, input->getContext());
  {
    auto dLdOutT = new NDArray(dLdOut->transpose());
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {dLdOutT, input};
    std::vector<NDArray*> outputs = {dLdOutTimesInput};
    mmulOp.execute(inputs, outputs);
    delete dLdOutT;
  }

  auto gradProd1 = new NDArray(*dLdOutTimesInput * *prod2);
  gradProd1->applyScalar(scalar::Multiply, scaling, gradProd1);
  auto gradProd2 = new NDArray(*dLdOutTimesInput * *prod1);
  gradProd2->applyScalar(scalar::Multiply, scaling, gradProd2);

  // dLdA1 = B1^T @ gradProd1
  {
    auto lohaB1T = new NDArray(lohaB1->transpose());
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB1T, gradProd1};
    std::vector<NDArray*> outputs = {dLdLohaA1};
    mmulOp.execute(inputs, outputs);
    delete lohaB1T;
  }

  // dLdB1 = gradProd1 @ A1^T
  {
    auto lohaA1T = new NDArray(lohaA1->transpose());
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {gradProd1, lohaA1T};
    std::vector<NDArray*> outputs = {dLdLohaB1};
    mmulOp.execute(inputs, outputs);
    delete lohaA1T;
  }

  // dLdA2 = B2^T @ gradProd2
  {
    auto lohaB2T = new NDArray(lohaB2->transpose());
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {lohaB2T, gradProd2};
    std::vector<NDArray*> outputs = {dLdLohaA2};
    mmulOp.execute(inputs, outputs);
    delete lohaB2T;
  }

  // dLdB2 = gradProd2 @ A2^T
  {
    auto lohaA2T = new NDArray(lohaA2->transpose());
    sd::ops::matmul mmulOp;
    std::vector<NDArray*> inputs = {gradProd2, lohaA2T};
    std::vector<NDArray*> outputs = {dLdLohaB2};
    mmulOp.execute(inputs, outputs);
    delete lohaA2T;
  }

  delete prod1;
  delete prod2;
  delete lohaDelta;
  delete dLdOutTimesInput;
  delete gradProd1;
  delete gradProd2;

  return Status::OK;
}

DECLARE_SHAPE_FN(loha_matmul_bp) {
  return SHAPELIST(
      CONSTANT(inputShape->at(0)),
      CONSTANT(inputShape->at(1)),
      CONSTANT(inputShape->at(2)),
      CONSTANT(inputShape->at(3)),
      CONSTANT(inputShape->at(4)),
      CONSTANT(inputShape->at(5))
  );
}

DECLARE_TYPES(loha_matmul_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
