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
// gather_bp — native backward pass for the gather op.
//
// Contract:
//   inputs[0]  = inputShapeVec  (1D integer array containing the forward input's shape)
//   inputs[1]  = indices        (integer; same indices as in forward gather)
//   inputs[2]  = gradOut        (upstream gradient; same shape as gather's output)
//   IArgs[0]   = axis           (same axis convention as forward gather)
//   output[0]  = dInput         (shape=inputShapeVec values, dtype=gradOut dtype,
//                                zero-init + scatter-add)
//
// The forward input's shape is passed as a runtime 1D integer array (input 0)
// rather than as IArgs.  This lets the op work for any input shape — including
// fully dynamic shapes that are not statically known at graph-build time.
// The shape vector's VALUES drive DECLARE_SHAPE_FN (value-dependent shape).
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gather_bp)

#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/gather_bp.h>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather_bp, 3, 1, false, 0, 1) {
  // inputs[0] = inputShapeVec  — 1D int array = shape of the original forward input
  //             (used ONLY by DECLARE_SHAPE_FN to allocate dInput; not read here)
  auto indices = INPUT_VARIABLE(1);   // gather indices   (ALL_INTS)
  auto gradOut = INPUT_VARIABLE(2);   // upstream gradient (ALL_FLOATS)

  auto dInput  = OUTPUT_VARIABLE(0);  // gradient w.r.t. input (shape from inputShapeVec)

  // IArgs: [0]=axis
  LongType axis = INT_ARG(0);
  const LongType inputRank = dInput->rankOf();
  if (axis < 0) axis += inputRank;

  REQUIRE_TRUE(axis >= 0 && axis < inputRank, 0,
               "GATHER_BP op: axis must be in [0, inputRank) after normalization; got %lld (rank %lld)",
               axis, inputRank);

  helpers::gatherBp(block.launchContext(), indices, gradOut, dInput, axis);

  return sd::Status::OK;
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(gather_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // inputShapeVec (forward input's shape as int array)
      ->setAllowedInputTypes(1, {ALL_INTS})     // indices
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradOut
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dInput
  // Output shape depends on the runtime VALUES of inputShapeVec, not just its shape.
  getOpDescriptor()->addTraits(OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}

////////////////////////////////////////////////////////////////////////
// Output shape is read from the VALUES of inputShapeVec (inputs[0]):
//   a 1D integer array whose elements are the forward input's shape dims.
// This is a value-dependent shape — mirrors the canonical fill.cpp pattern:
//   shapeArray->e<LongType>(i) reads each dim from the runtime array.
// dtype comes from gradOut (inputs[2]).
DECLARE_SHAPE_FN(gather_bp) {
  // inputShapeVec is the runtime shape array (input 0).
  // Use INPUT_VARIABLE to get the NDArray whose VALUES are the forward input dims.
  auto inputShapeVec = INPUT_VARIABLE(0);
  const LongType rank = inputShapeVec->lengthOf();
  std::vector<LongType> dims(rank);
  for (LongType i = 0; i < rank; ++i)
    dims[i] = inputShapeVec->e<LongType>(i);

  // gradOut is inputs[2]; its dtype becomes the output dtype
  auto gradOutType = ArrayOptions::dataType(inputShape->at(2));

  auto result = ConstantShapeHelper::getInstance().createShapeInfo(gradOutType, 'c', dims);
  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_gather_bp)
