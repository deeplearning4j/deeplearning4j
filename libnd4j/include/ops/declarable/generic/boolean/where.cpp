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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_Where)

#include <helpers/ShapeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <array/ArrayOptions.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {

// Helper function to evaluate condition regardless of underlying data type
inline bool evaluateCondition(NDArray* condition, int index) {
 switch(condition->dataType()) {
#if defined(HAS_BOOL)
   case DataType::BOOL:
     // Read BOOL as int8 and compare to 0 for consistent behavior
     return condition->e<int8_t>(index) != 0;
#endif
#if defined(HAS_INT8)
   case DataType::INT8:
     return condition->e<int8_t>(index) != 0;
#endif
#if defined(HAS_INT16)
   case DataType::INT16:
     return condition->e<int16_t>(index) != 0;
#endif
#if defined(HAS_INT32)
   case DataType::INT32:
     return condition->e<int32_t>(index) != 0;
#endif
#if defined(HAS_LONG)
   case DataType::INT64:
     return condition->e<sd::LongType>(index) != 0;
#endif
#if defined(HAS_UINT8)
   case DataType::UINT8:
     return condition->e<uint8_t>(index) != 0;
#endif
#if defined(HAS_UINT16)
   case DataType::UINT16:
     return condition->e<uint16_t>(index) != 0;
#endif
#if defined(HAS_UINT32)
   case DataType::UINT32:
     return condition->e<uint32_t>(index) != 0;
#endif
#if defined(HAS_UNSIGNEDLONG)
   case DataType::UINT64:
     return condition->e<sd::UnsignedLong>(index) != 0;
#endif
#if defined(HAS_FLOAT16)
   case DataType::HALF:
     return condition->e<float16>(index) != static_cast<float16>(0.0f);
#endif
#if defined(HAS_BFLOAT16)
   case DataType::BFLOAT16:
     return condition->e<bfloat16>(index) != static_cast<bfloat16>(0.0f);
#endif
#if defined(HAS_FLOAT32)
   case DataType::FLOAT32:
     return condition->e<float>(index) != 0.0f;
#endif
#if defined(HAS_DOUBLE)
   case DataType::DOUBLE:
     return condition->e<double>(index) != 0.0;
#endif
   default:
     // Fallback: try to interpret as int32 and check if non-zero
#if defined(HAS_INT32)
#ifdef __cpp_exceptions
     try {
       return condition->e<int32_t>(index) != 0;
     } catch (...) {
       // Last resort: assume false to maintain safe behavior
       return false;
     }
#else
     return condition->e<int32_t>(index) != 0;
#endif
#else
     // If INT32 is not available, return false as safe default
     return false;
#endif
 }
}

CUSTOM_OP_IMPL(Where, 1, 1, false, 0, 0) {
 auto condition = INPUT_VARIABLE(0);
 auto z = OUTPUT_VARIABLE(0);
 if (z->isEmpty()) return Status::OK;

 if (block.width() == 3) {
   auto x = INPUT_VARIABLE(1);
   auto y = INPUT_VARIABLE(2);

   // Check if x and y can be broadcast together (instead of requiring exact same shape)
   REQUIRE_TRUE(x->isSameShape(y) || ShapeUtils::areShapesBroadcastable(*x, *y), 0,
                "X and Y must have equal shapes or be broadcastable. X shape: %s, Y shape: %s",
                ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

   // Case 1: All arrays have exact shape matching or are broadcastable (element-wise operation)
   if ((condition->isSameShape(x) && x->isSameShape(y)) ||
       (ShapeUtils::areShapesBroadcastable(*condition, *x) &&
        ShapeUtils::areShapesBroadcastable(*condition, *y) &&
        ShapeUtils::areShapesBroadcastable(*x, *y))) {
     // Use GPU-accelerated helper for element-wise where
     helpers::_whereElementWise(block.launchContext(), *condition, *x, *y, *z);
   }
   // Case 2: TAD-mask operation (condition is 1D, selecting entire TADs)
   else if (condition->rankOf() == 1 && condition->lengthOf() == x->sizeAt(0)) {
     std::vector<LongType> axis({0});
     helpers::_whereTad(block.launchContext(), *condition, *x, *y, *z, axis);
   }
   // Case 3: Invalid shapes - provide detailed error message
   else {
     std::string condShape = ShapeUtils::shapeAsString(condition);
     std::string xShape = ShapeUtils::shapeAsString(x);
     std::string yShape = ShapeUtils::shapeAsString(y);

     REQUIRE_TRUE(false, 0,
                  "Where operation: Invalid shapes for broadcasting. "
                  "Condition shape: %s, X shape: %s, Y shape: %s. "
                  "Condition must either: (1) match X/Y shapes exactly, "
                  "(2) be broadcastable with X/Y shapes, or "
                  "(3) be 1D with length equal to first dimension of X/Y for TAD-mask operation.",
                  condShape.c_str(), xShape.c_str(), yShape.c_str());
   }
 } else {
   // in this case we return 2D matrix, which basically contains coordinates of true elements
   REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead",
                block.width());
   auto output = OUTPUT_VARIABLE(0);

   if (z->isEmpty()) return Status::OK;

   helpers::_where(block.launchContext(), *condition, *output, block.workspace());
 }
 return Status::OK;
}

DECLARE_SHAPE_FN(Where) {
 if (block.width() == 3) {
   auto x = INPUT_VARIABLE(1);
   auto y = INPUT_VARIABLE(2);

   // Calculate the broadcast result shape for x and y
   LongType* resultShapeInfo = nullptr;
   bool canBroadcast = ShapeUtils::evalBroadcastShapeInfo(*x, *y, true, resultShapeInfo, block.getWorkspace());

   if (canBroadcast && resultShapeInfo != nullptr) {
     return SHAPELIST(CONSTANT(resultShapeInfo));
   } else {
     // Fallback to x's shape if broadcasting fails (should have been caught in validation)
     auto inShape = inputShape->at(1);
     return SHAPELIST(CONSTANT(inShape));
   }
 } else {
   // output shape is the 2D tensor num_true x rankOf (inShape)
   auto condition = INPUT_VARIABLE(0);
   auto inShape = inputShape->at(0);

   // Sync condition to host before accessing data via e<T>()
   condition->syncToHost();

   LongType numOfTrue = 0;  // condition->reduceNumber(reduce::CountNonZero, nullptr).e<sd::LongType>(0);

   // Debug: print condition info
   sd_debug("Where shape function: condition shape=%s, dtype=%d, length=%lld\n",
            ShapeUtils::shapeAsString(condition).c_str(),
            static_cast<int>(condition->dataType()),
            condition->lengthOf());

   for (LongType i = 0; i < condition->lengthOf(); i++)
     if (evaluateCondition(condition, i)) numOfTrue++;

   // Sync back to device so subsequent GPU kernels see current device buffer
   condition->syncToDevice();

   sd_debug("Where shape function: found %lld true values out of %lld total\n", numOfTrue, condition->lengthOf());

   LongType * theNewShape;
   LongType conditionRank = shape::rank(inShape);

   if (numOfTrue == 0) {
     // For empty result, use emptyShapeInfoWithShape which properly sets up empty arrays
     std::vector<LongType> emptyShape = {0, conditionRank};
#if defined(HAS_LONG)
     theNewShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(INT64, emptyShape);
#else
     theNewShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(INT32, emptyShape);
#endif
   } else {
     // For non-empty result, create shape [numOfTrue, conditionRank]
     LongType* newShape;
     ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
     newShape[0] = 2;  // rank
     newShape[1] = numOfTrue;  // rows (number of true elements)
     newShape[2] = conditionRank;  // cols (coordinates per true element)
     newShape[3] = conditionRank;  // stride for dim 0
     newShape[4] = 1;  // stride for dim 1
     newShape[5] = 0;  // offset
     newShape[6] = 1;  // ews
     newShape[7] = 99; // order 'c'
#if defined(HAS_LONG)
     ShapeUtils::updateStridesAndType(newShape, INT64, 'c');
#else
     ShapeUtils::updateStridesAndType(newShape, INT32, 'c');
#endif
     theNewShape = CONSTANT(newShape);
     RELEASE(newShape, block.getWorkspace());
   }

   return SHAPELIST(theNewShape);
 }
}

DECLARE_TYPES(Where) {
 getOpDescriptor()
     ->setAllowedInputTypes(0, ANY)  // bool
     ->setAllowedInputTypes(1, ANY)
     ->setAllowedInputTypes(2, ANY)
     ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS,BOOL});
 getOpDescriptor()->addTraits(OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT);
}
}  // namespace ops
}  // namespace sd

#endif