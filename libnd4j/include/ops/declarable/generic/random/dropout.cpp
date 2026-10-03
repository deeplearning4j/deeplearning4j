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
// Created by GS <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dropout)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/dropout.h>
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout, 1, 2, true, 1, 1) {
  auto input = INPUT_VARIABLE(0);  // lookup param
  bool inverted = block.numB() > 0 ? B_ARG(0) : false;
  NDArray* reduceShape = nullptr;  // this param is optional
  // The helpers write every element of both outputs (a dropped element is stored as 0 and the mask records every
  // decision), so neither is pre-zeroed. Zeroing an output that aliases the input (the in-place form) would erase
  // the input before it is read.
  auto output = OUTPUT_VARIABLE(0);
  auto mask = OUTPUT_VARIABLE(1);

  int seed = INT_ARG(0);

  // probValue is the probability of KEEPING an element. The probability argument is that keep probability, or, with
  // the inverted flag set, the probability of dropping (probValue = 1 - argument).
  double probValue = T_ARG(0);
  if(inverted) {
    probValue = 1 - probValue;
  }


  REQUIRE_TRUE(probValue >= 0.f && probValue <= 1.f, 0, "dropout: Probability should be with range 0 to 1.");

  if (probValue == 1.0f) {
    *output = *input;
    double one = 1.0;
    mask->assign(one);
    return Status::OK;
  }

  return helpers::dropOutFunctor(block, input, output, reduceShape, seed, probValue, mask);
}

DECLARE_TYPES(dropout) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS,ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setSameMode(true);

  // Unseeded, the mask is drawn from the context's random generator.
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_STATEFUL);
}

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout_bp, 3, 1, false, 1, 1) {
  auto input = INPUT_VARIABLE(0);    // lookup param
  auto mask = INPUT_VARIABLE(1);
  auto gradOut = INPUT_VARIABLE(2);  // lookup param
  bool inverted = block.numB() > 0 ? B_ARG(0) : false;

  NDArray* reduceShape = nullptr;         // this param is optional
  // The gradient writes every output element (gradOut * mask), so the output is not pre-zeroed: it may alias gradOut
  // (the in-place form), and zeroing it would erase the gradient before it is read.
  auto output = OUTPUT_VARIABLE(0);

  int seed = INT_ARG(0);

  // Same probability convention as the forward: the keep probability, or 1 - argument with the inverted flag set.
  double probValue = T_ARG(0);
  if(inverted) {
    probValue = 1 - probValue;
  }

  REQUIRE_TRUE((probValue >= 0. && probValue <= 1.), 0, "dropout_bp: Probability should be with range 0 to 1.");

  // No special case for probValue == 1 (nothing dropped) or 0 (everything dropped): the mask the forward produced
  // records exactly which elements it kept (all ones, respectively all zeros), and gradOut * mask is the gradient in
  // every case. A keep probability of 1 must pass the gradient through, not zero it.
  REQUIRE_TRUE(sd::ops::helpers::dropOutFunctorBP(block, input, gradOut, output, reduceShape, seed, probValue,
                                                  mask) == sd::Status::OK,
               0, "dropout_bp: Cannot backprop dropout.");

  return Status::OK;
}

DECLARE_TYPES(dropout_bp) {

  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})->setAllowedOutputTypes({ALL_FLOATS});

  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(alpha_dropout_bp, 2, 1, false, 4, 1) {
  NDArray* input = INPUT_VARIABLE(0);    // lookup param
  NDArray *mask = INPUT_VARIABLE(1);     // lookup param
  NDArray* gradOut = INPUT_VARIABLE(2);  // lookup param

  NDArray* reduceShape = nullptr;        // this param is optional
  NDArray* output = OUTPUT_VARIABLE(0);  //

  int seed = INT_ARG(0);

  double probValue = T_ARG(0);
  double alphaValue = T_ARG(1);
  double alpha1Value = T_ARG(2);
  double betaValue = T_ARG(3);

  REQUIRE_TRUE(probValue > 0. && probValue <= 1., 0, "dropout_bp: Probability should be with range 0 to 1.");
  if (probValue == 1.0) {
    double zero = 0.0;
    output->assign(zero);  // fill up output with 0
    return Status::OK;
  }

  return helpers::alphaDropOutFunctorBP(block, input, gradOut, output, reduceShape, seed, probValue, alphaValue,
                                        alpha1Value, betaValue, mask);
}
DECLARE_TYPES(alpha_dropout_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION | OP_TRAIT_BACKWARD);
}
}  // namespace ops
}  // namespace sd

#endif
