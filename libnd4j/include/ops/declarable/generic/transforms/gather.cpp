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
// @author Shyrma Yurii (iuriish@yahoo.com), created on 16.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gather)

#include <graph/DspLifecycleContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/gather.h>
#include <ops/declarable/helpers/scatter.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto indices = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  // Skip index validation during DSP capture/replay — indices were validated during warmup
  const bool checkIndices = graph::DspLifecycleContext::isOwned()
      ? false
      : (block.getBArguments()->empty() ? true : B_ARG(0));

  // Edge case: empty indices or empty input -> empty output
  bool indicesEmpty = indices != nullptr && (indices->isEmpty() || indices->lengthOf() == 0);
  bool inputEmpty = input->isEmpty() || input->lengthOf() == 0;
  bool outputEmpty = output->isEmpty() || output->lengthOf() == 0;

  if (indicesEmpty || inputEmpty || outputEmpty) {
    // For empty arrays, just return - nothing to gather
    return sd::Status::OK;
  }

  const sd::LongType numOfIntArgs = block.numI();

  std::vector<sd::LongType> intArgs;
  if (block.width() > 2) {
    intArgs = INPUT_VARIABLE(2)->template asVectorT<sd::LongType>();
  } else {
    if (numOfIntArgs == 0) {
      intArgs.emplace_back(0);
    } else {
      intArgs.reserve(numOfIntArgs);
      auto iArgs = block.getIArguments();
      for (sd::LongType i = 0; i < numOfIntArgs; i++) {
        intArgs.emplace_back(iArgs->at(i));
      }
    }
  }

  const sd::LongType inputRank = input->rankOf();
  if (intArgs[0] < 0) intArgs[0] += inputRank;

  // input validation
  REQUIRE_TRUE(intArgs[0] >= 0, 0,
               "GATHER op: input axis must be non-negative after normalization, but got %i!", intArgs[0]);
  REQUIRE_TRUE(intArgs[0] < inputRank, 0,
               "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!",
               intArgs[0], inputRank);
  REQUIRE_TRUE(indices != nullptr || numOfIntArgs > 1, 0,
               "GATHER op: indices should be provided either as additional input array or as IntArguments !");

  if (checkIndices) {
    NDArray* pIndices = indices;
    bool ownsIndices = false;
    
    if (indices == nullptr) {
      std::vector<sd::LongType> shape = {static_cast<sd::LongType>(intArgs.size()) - 1};
      std::vector<double> inputVec = std::vector<double>(intArgs.begin() + 1, intArgs.end());
      pIndices = new NDArray(input->ordering(), shape, inputVec, DataType::INT64, block.launchContext());
      ownsIndices = true;
    }
    
    const sd::LongType numOfBadIndx = helpers::checkIndices(block.launchContext(), *pIndices, *input, intArgs[0]);

    // FIXED: Cleanup BEFORE checking condition (REQUIRE_TRUE can throw)
    if (ownsIndices) {
      delete pIndices;
      pIndices = nullptr;
    }

    // Diagnostic: dump actual index values when OOB detected
    if (numOfBadIndx > 0) {
      sd::LongType axis = intArgs[0];
      sd::LongType dimSize = input->sizeAt(axis);
      // Sync indices to host for reading
      if (indices != nullptr) {
        indices->syncToHost();
        sd_printf("GATHER OOB DIAGNOSTIC: axis=%lld dimSize=%lld indicesShape=", axis, dimSize);
        for (int r = 0; r < indices->rankOf(); r++) {
          sd_printf("%s%lld", r > 0 ? "x" : "", indices->sizeAt(r));
        }
        sd_printf(" dtype=%d\n", (int)indices->dataType());
        // Also print full stride info for indices
        sd_printf("GATHER OOB STRIDES: indicesStrides=[");
        const sd::LongType* _istrides = indices->stridesOf();
        for (int r = 0; r < indices->rankOf(); r++) {
          sd_printf("%s%lld", r > 0 ? "," : "", (long long)_istrides[r]);
        }
        sd_printf("] order=%c offset=%lld bufAddr=%p\n", indices->ordering(),
                  (long long)indices->offset(),
                  indices->dataBuffer() ? indices->dataBuffer()->primary() : nullptr);
        // Print first 32 index values to understand pattern
        sd::LongType dumpCount = std::min(indices->lengthOf(), (sd::LongType)32);
        for (sd::LongType idx = 0; idx < dumpCount; idx++) {
          auto val = indices->e<sd::LongType>(idx);
          sd_printf("  indices[%lld] = %lld %s\n", idx, val,
                    (val < 0 || val >= dimSize) ? "<-- OOB" : "");
        }
        // Also print input tensor shape/stride for context
        sd_printf("GATHER OOB INPUT: inputShape=[");
        for (int r = 0; r < input->rankOf(); r++) {
          sd_printf("%s%lld", r > 0 ? "x" : "", (long long)input->sizeAt(r));
        }
        sd_printf("] inputStrides=[");
        const sd::LongType* _in_strides = input->stridesOf();
        for (int r = 0; r < input->rankOf(); r++) {
          sd_printf("%s%lld", r > 0 ? "," : "", (long long)_in_strides[r]);
        }
        sd_printf("] order=%c offset=%lld\n", input->ordering(), (long long)input->offset());
      }
    }

    // Check condition after cleanup
    REQUIRE_TRUE(numOfBadIndx == 0, 0,
                 "GATHER OP: please check elements of indices-array, total number of wrong elements is %lld!",
                 numOfBadIndx);
  }

  helpers::gather(block.launchContext(), input, indices, output, intArgs);

  return sd::Status::OK;
}

DECLARE_TYPES(gather) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS, BOOL});
  getOpDescriptor()->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS, BOOL});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_GATHER);
}

DECLARE_SHAPE_FN(gather) {
  // check shape of paddings
  auto inputShapeInfo = inputShape->at(0);
  sd::LongType* outputShapeInfo = nullptr;

  sd::LongType axis = 0;

  if (block.width() > 2) {
    axis = INPUT_VARIABLE(2)->e<sd::LongType>(0);
  } else
    axis = block.numI() > 0 ? block.getIArguments()->at(0) : 0;

  sd::LongType inputRank = shape::rank(inputShapeInfo);
  if (axis < 0) axis += inputRank;

  REQUIRE_TRUE(axis >= 0, 0,
               "GATHER op: input axis must be non-negative after normalization, but got %i!", axis);
  REQUIRE_TRUE(axis < inputRank, 0,
               "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!", axis,
               inputRank);

  bool isEmpty = false;

  if (block.width() > 1) {
    auto indicesShapeInfo = inputShape->at(1);

    sd::LongType indicesRank = shape::rank(indicesShapeInfo);

    sd::LongType outputRank = inputRank + indicesRank - 1;

    // Special handling for scalar output (rank 0)
    if (outputRank == 0) {
      auto result = ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inputShapeInfo));
      return SHAPELIST(result);
    }

    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), sd::LongType);

    // fill output shapeInfo
    outputShapeInfo[0] = outputRank;
    sd::LongType shapeIdx = 1;

    for (sd::LongType i = 0; i < axis; ++i) outputShapeInfo[shapeIdx++] = inputShapeInfo[i + 1];

    for (sd::LongType i = 0; i < indicesRank; ++i) outputShapeInfo[shapeIdx++] = indicesShapeInfo[i + 1];

    for (sd::LongType i = axis + 1; i < inputRank; ++i) outputShapeInfo[shapeIdx++] = inputShapeInfo[i + 1];
  } else if (block.numI() > 1) {
    int indicesRank = block.numI() == 2 ? 0 : 1;

    sd::LongType outputRank = inputRank + indicesRank - 1;

    // Special handling for scalar output (rank 0)
    if (outputRank == 0) {
      auto result = ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inputShapeInfo));
      return SHAPELIST(result);
    }

    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), sd::LongType);

    // building shape manually
    outputShapeInfo[0] = outputRank;
    int shapeIdx = 1;
    for (sd::LongType i = 0; i < axis; ++i) outputShapeInfo[shapeIdx++] = inputShapeInfo[i + 1];

    if (block.numI() > 2) outputShapeInfo[shapeIdx++] = block.numI() - 1;

    for (sd::LongType i = axis + 1; i < inputRank; ++i) outputShapeInfo[shapeIdx++] = inputShapeInfo[i + 1];
  } else
    REQUIRE_TRUE(false, 0,
                 "GATHER op: indices should be provided either as additional input array or as IntArguments !");

  ShapeUtils::updateStridesAndType(outputShapeInfo, inputShapeInfo, shape::order(inputShapeInfo));

  // Check if output has any zero dimensions (making it empty)
  if (shape::length(outputShapeInfo) == 0) {
    ArrayOptions::setPropertyBit(outputShapeInfo, ARRAY_EMPTY);
  }

  auto result = ConstantShapeHelper::getInstance().bufferForShapeInfo(outputShapeInfo)->primary();
  RELEASE(outputShapeInfo, block.getWorkspace());
  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd
#endif
