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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_permute)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// here iArgs is int vector of ordered set of dimensions to be permuted
CUSTOM_OP_IMPL(permute, 1, 1, true, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "PERMUTE OP: when input is empty, output must also be empty");
    return Status::OK;  // No op
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    z->assign(x->transpose());
    return Status::OK;
  }

  std::vector<LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<LongType>() : *block.getIArguments();
  if(permutationVector.size() != x->rankOf()) {
    sd_printf("PERMUTE OP: permutation vector size was %d and x input rank was %d\n",permutationVector.size(),x->rankOf());
  }
  REQUIRE_TRUE(permutationVector.size() == x->rankOf(),permutationVector.size(),"PERMUTE OP: number of permutations is less in size than input rank.");

  z->assign(x->permute(permutationVector));

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(permute) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedInputTypes(1, {ALL_INTS})->setSameMode(true);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(permute) {
  auto x = INPUT_VARIABLE(0);

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    return SHAPELIST(ShapeUtils::evalTransposeShapeInfo(*x, block.workspace(), true));
  }
  std::vector<LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<LongType>() : *block.getIArguments();

  auto outputShapeInfo =
      ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, block.workspace(), true);
  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
