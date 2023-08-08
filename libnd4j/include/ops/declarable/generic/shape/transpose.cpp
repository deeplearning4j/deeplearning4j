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
#if NOT_EXCLUDED(OP_transpose)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(transpose, 1, 1, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // Special case: empty.reshape(<other empty shape>) -> return empty
  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "TRANSPOSE OP: when input is empty, output must also be empty");
    return sd::Status::OK;  // No op
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    z->assign(x->transpose());
    return sd::Status::OK;
  }

  std::vector<sd::LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<sd::LongType>() : *block.getIArguments();

  z->assign(x->permute(permutationVector));

  return sd::Status::OK;
}

DECLARE_TYPES(transpose) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(transpose) {
  auto x = INPUT_VARIABLE(0);

  if (block.width() == 1 && block.getIArguments()->size() == 0)
    return SHAPELIST(ShapeUtils::evalTransposeShapeInfo(*x, block.workspace(), true));

  std::vector<sd::LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<sd::LongType>() : *block.getIArguments();
  bool isPermuteNecessary = false;
  const sd::LongType rank = x->rankOf();
  for (sd::LongType i = 0; i < rank; ++i) {
    if (permutationVector[i] != i) {
      isPermuteNecessary = true;
      break;
    }
  }

  if(!isPermuteNecessary) {
        auto outputShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(const_cast<sd::LongType *>(x->shapeInfo()),true);
        return SHAPELIST(outputShapeInfo);
  }


  //TODO: likely issue we need to sort out with cuda and data here. Change this to be a proper vector and
  //debug why this is corrupt.
  auto outputShapeInfo =
      ConstantShapeHelper::getInstance().createFromExisting(const_cast<sd::LongType *>(ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), *x, block.workspace(), true)),true);

  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
