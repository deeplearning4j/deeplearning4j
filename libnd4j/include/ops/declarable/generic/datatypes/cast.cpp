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
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_cast)

#include <array/DataTypeUtils.h>
#include <ops/declarable/headers/datatypes.h>
#include <ops/declarable/helpers/assign.h>
namespace sd {
namespace ops {
static bool isCastStorageType(DataType type) {
  return DataTypeUtils::isR(type) || DataTypeUtils::isB(type) ||
         (DataTypeUtils::validDataType(type) && !DataTypeUtils::isS(type));
}

// Validate before narrowing an IArg or allocating a shape; enum sentinels are not storage types.
static DataType castTargetType(graph::Context& block) {
  const auto& integers = *block.getIArguments();
  const auto& types = *block.getDArguments();
  REQUIRE_TRUE(integers.size() <= 1 && types.size() <= 1 && (!integers.empty() || !types.empty()), 0,
               "cast: expected one target dtype in IArgs or DArgs");
  DataType target = UNKNOWN;
  if (!integers.empty()) {
    const LongType code = integers[0];
    REQUIRE_TRUE(code >= 0 && code <= 255, 0, "cast: target dtype code is out of range");
    target = DataTypeUtils::fromInt(static_cast<int>(code));
    REQUIRE_TRUE(isCastStorageType(target), 0, "cast: target must be numeric or BOOL");
  }
  if (!types.empty()) {
    REQUIRE_TRUE(isCastStorageType(types[0]), 0, "cast: target must be numeric or BOOL");
    REQUIRE_TRUE(integers.empty() || target == types[0], 0, "cast: IArg and DArg targets disagree");
    target = types[0];
  }
  return target;
}

CUSTOM_OP_IMPL(cast, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  const auto target = castTargetType(block);
  REQUIRE_TRUE(isCastStorageType(input->dataType()), 0, "cast: input must be numeric or BOOL");
  REQUIRE_TRUE(output->dataType() == target && output->isSameShape(input), 0,
               "cast: output must have the requested dtype and input shape");
  REQUIRE_TRUE(!block.isInplace() || input->dataType() == target, 0,
               "cast: dtype-changing casts cannot execute in place");

  if (input->isEmpty()) {
    REQUIRE_TRUE(output->isEmpty(), 0, "If input is empty, output array must also be empty");
    return sd::Status::OK;
  }

  // Fast path: same data type - no conversion needed
  if (input->dataType() == output->dataType()) {
    if (!block.isInplace()) {
      output->assign(input);
    }
  } else if (!block.isInplace()) {
    helpers::assign(block.launchContext(), output, input);
  }

  STORE_RESULT(output);
  return sd::Status::OK;
}
DECLARE_SYN(Cast, cast);

samediff::EmptyHandling SD_BACKEND_OPS_CLASS(cast)::emptyHandling() {
  return samediff::EmptyHandling::EMPTY_EXECUTE;
}

DECLARE_SHAPE_FN(cast) {
  auto inShape = inputShape->at(0);
  const auto newType = castTargetType(block);
  REQUIRE_TRUE(isCastStorageType(ArrayOptions::dataType(inShape)), 0,
               "cast: input must be numeric or BOOL");

  // Check empty from both native shape info AND from the NDArray object.
  // Java-created empty singletons (Nd4j.empty()) may lack the ARRAY_EMPTY bit
  // in the native C++ shape pointer even though isEmpty() returns true on the Java side.
  // NDArray::isEmpty() has a fallback: rank==0 && _buffer==nullptr catches these.
  bool wasEmptyFromShape = ArrayOptions::hasPropertyBitSet(inShape, ARRAY_EMPTY);
  bool wasEmptyFromArray = false;
  if (block.isFastPath()) {
    const auto& fp = block.fastpath_in();
    if (fp.size() > 0 && fp[0] != nullptr) wasEmptyFromArray = fp[0]->isEmpty();
  }
  bool wasEmpty = wasEmptyFromShape || wasEmptyFromArray;

  if (wasEmpty) {
    auto desc = ShapeBuilders::emptyShapeInfo(newType, shape::order(inShape), shape::rank(inShape),
                                             shape::shapeOf(inShape));
    auto result = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
    delete[] desc;
    return result;
  }

  // Both argument encodings allocate the same dense output, independently of input view strides.
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(newType, inShape));
}

DECLARE_TYPES(cast) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS, BOOL, FLOAT8, FLOAT8_E5M2})
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS, BOOL, FLOAT8, FLOAT8_E5M2});
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_CAST);
}
}  // namespace ops
}  // namespace sd

#endif
