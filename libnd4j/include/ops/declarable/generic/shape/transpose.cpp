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
#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(transpose, 1, 1, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // Special case: empty.reshape(<other empty shape>) -> return empty
  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "TRANSPOSE OP: when input is empty, output must also be empty");
    return Status::OK;  // No op
  }

  NDArray* castedPermute = nullptr;
  std::vector<LongType> permutationVector;
  if (block.width() > 1) {
    // Read permutation indices directly from synced host buffer to avoid e<T>() bugs on CUDA.
    auto* permArr = INPUT_VARIABLE(1);
    permArr->syncToHost();
    auto* db = permArr->dataBuffer();
    auto numEl = permArr->lengthOf();
    auto dtype = permArr->dataType();
    auto arrOffset = permArr->offset();
    if (dtype == INT64) {
      auto* hostLongs = reinterpret_cast<LongType*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(hostLongs[i]);
    } else if (dtype == INT32) {
      auto* hostInts = reinterpret_cast<int*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(static_cast<LongType>(hostInts[i]));
    } else {
      castedPermute = permArr->cast(INT64);
      permutationVector = castedPermute->asVectorT<LongType>();
    }
  } else {
    permutationVector = *block.getIArguments();
  }

  if (permutationVector.size() == 0) {
    NDArray *t = x->transpose();
    z->assign(t);
    // FIXED: transpose() returns a view - only delete if not a view
    if (t != nullptr && !t->isView()) {
      delete t;
    }
    if (castedPermute != nullptr) delete castedPermute;
    return Status::OK;
  }

  bool isPermuteNecessary = false;

  int rank = permutationVector.size();
  //handles empty permute vector case as well as case where array rank and permute vector rank
  //are different
  for (LongType i = 0; i < rank; ++i) {
    if (permutationVector[i] != i) {
      isPermuteNecessary = true;
      break;
    }
  }
  if(!isPermuteNecessary) {
    z->assign(x);
    return Status::OK;
  }

  z->assign(x->permute(permutationVector, false, false));

  if (castedPermute != nullptr) delete castedPermute;
  return Status::OK;
}

DECLARE_TYPES(transpose) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(transpose) {
  auto x = INPUT_VARIABLE(0);
  const LongType rank = x->rankOf();

  if(rank < 1)
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(x->dataType()));
  std::vector<LongType> permutationVector;
  if (block.width() > 1) {
    auto* permArr = INPUT_VARIABLE(1);
    permArr->syncToHost();
    auto* db = permArr->dataBuffer();
    auto numEl = permArr->lengthOf();
    auto dtype = permArr->dataType();
    auto arrOffset = permArr->offset();
    if (dtype == INT64) {
      auto* hostLongs = reinterpret_cast<LongType*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(hostLongs[i]);
    } else if (dtype == INT32) {
      auto* hostInts = reinterpret_cast<int*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(static_cast<LongType>(hostInts[i]));
    } else {
      auto* casted = permArr->cast(INT64);
      permutationVector = casted->asVectorT<LongType>();
      delete casted;
    }
  } else {
    permutationVector = *block.getIArguments();
  }

  if (permutationVector.size() == 0) {
    auto temp = ShapeUtils::evalTransposeShapeInfo(*x, nullptr, true);
    auto ret = ConstantShapeHelper::getInstance().createFromExisting(temp);
    RELEASE(temp, nullptr);
    return SHAPELIST(ret);
  }


  bool isPermuteNecessary = false;

  if(permutationVector.size() == static_cast<size_t>(rank))
    for (LongType i = 0; i < rank; ++i) {
      if (permutationVector[i] != i) {
        isPermuteNecessary = true;
        break;
      }
    }

  if(!isPermuteNecessary) {
    //note: do not deallocate this buffer. they are kept around.
    auto permEvalShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(inputShape->at(0));
    return SHAPELIST(permEvalShapeInfo);
  }


  if(x->isEmpty()) {
    // For empty arrays, create shape with ARRAY_EMPTY flag set
    auto permEvalShapeInfo = ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, nullptr, false);
    ArrayOptions::setPropertyBit(permEvalShapeInfo, ARRAY_EMPTY);
    auto ret = ConstantShapeHelper::getInstance().bufferForShapeInfo(permEvalShapeInfo)->primary();
    RELEASE(permEvalShapeInfo, nullptr);
    return SHAPELIST(ret);
  }

  // Non-empty case: use cached shape directly
  auto permEvalShapeInfo = ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, nullptr, true);
  auto ret = ConstantShapeHelper::getInstance().createFromExisting(permEvalShapeInfo);
  RELEASE(permEvalShapeInfo, nullptr);
  return SHAPELIST(ret);
}

}  // namespace ops
}  // namespace sd

#endif
