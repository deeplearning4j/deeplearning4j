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
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {

// Helper function to evaluate condition regardless of underlying data type
inline bool evaluateCondition(NDArray* condition, int index) {
 switch(condition->dataType()) {
#if defined(HAS_BOOL)
   case DataType::BOOL:
     return condition->e<bool>(index);
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
     return condition->e<uint64_t>(index) != 0;
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
     try {
       return condition->e<int32_t>(index) != 0;
     } catch (...) {
       // Last resort: assume false to maintain safe behavior
       return false;
     }
#else
     // If INT32 is not available, return false as safe default
     return false;
#endif
 }
}

// Helper function to perform element-wise where with proper broadcasting
void performBroadcastedWhere(NDArray* condition, NDArray* x, NDArray* y, NDArray* z) {
 // We'll process each element of the output array z
 // and determine the appropriate indices for condition, x, and y based on broadcasting rules

 auto zShape = z->getShapeAsVector();
 auto condShape = condition->getShapeAsVector();
 auto xShape = x->getShapeAsVector();
 auto yShape = y->getShapeAsVector();

 // For each element in the output array
 for (LongType i = 0; i < z->lengthOf(); i++) {
   // Convert linear index to multi-dimensional indices for output array
   std::vector<LongType> zIndices(z->rankOf());
   LongType remainder = i;
   for (int dim = z->rankOf() - 1; dim >= 0; dim--) {
     zIndices[dim] = remainder % z->sizeAt(dim);
     remainder /= z->sizeAt(dim);
   }

   // Calculate corresponding indices in condition, x, and y arrays using broadcasting rules
   auto getLinearIndex = [](const std::vector<LongType>& multiIndices, const std::vector<LongType>& shape, NDArray* array) -> LongType {
     LongType linearIndex = 0;
     LongType stride = 1;
     int srcDim = shape.size() - 1;

     for (int dim = multiIndices.size() - 1; dim >= 0; dim--) {
       LongType srcIndex = 0;
       if (srcDim >= 0) {
         if (shape[srcDim] == 1) {
           srcIndex = 0; // Broadcast dimension
         } else {
           srcIndex = multiIndices[dim];
         }
         srcDim--;
       }
       linearIndex += srcIndex * stride;
       if (srcDim >= 0) {
         stride *= shape[srcDim + 1];
       }
     }
     return linearIndex;
   };

   LongType condIndex = condition->lengthOf() == 1 ? 0 : getLinearIndex(zIndices, condShape, condition);
   LongType xIndex = x->lengthOf() == 1 ? 0 : getLinearIndex(zIndices, xShape, x);
   LongType yIndex = y->lengthOf() == 1 ? 0 : getLinearIndex(zIndices, yShape, y);

   // Apply the where logic
   if (z->isR()) {
     auto result = evaluateCondition(condition, condIndex) ?
                                                           x->e<double>(xIndex) : y->e<double>(yIndex);
     z->p(i, result);
   } else {
     auto result = evaluateCondition(condition, condIndex) ?
                                                           x->e<LongType>(xIndex) : y->e<LongType>(yIndex);
     z->p(i, result);
   }
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

   // Case 1: All arrays have exact shape matching (element-wise operation)
   if (condition->isSameShape(x) && x->isSameShape(y)) {
     // FIXME: for perf it might be better to issue memcpy here, and fill only mismatched values from either X or Y
     for (int e = 0; e < condition->lengthOf(); e++) {
       if (z->isR()) {
         auto r = !evaluateCondition(condition, e) ? y->e<double>(e) : x->e<double>(e);
         z->p(e, r);
       } else {
         auto r = !evaluateCondition(condition, e) ? y->e<LongType>(e) : x->e<LongType>(e);
         z->p(e, r);
       }
     }
   }
   // Case 2: Broadcasting is possible (most flexible case)
   else if (ShapeUtils::areShapesBroadcastable(*condition, *x) &&
            ShapeUtils::areShapesBroadcastable(*condition, *y) &&
            ShapeUtils::areShapesBroadcastable(*x, *y)) {
     performBroadcastedWhere(condition, x, y, z);
   }
   // Case 3: TAD-mask operation (legacy behavior for specific cases)
   else if (condition->rankOf() == 1 && condition->lengthOf() == x->sizeAt(0)) {
     std::vector<LongType> zero({0});
     auto dims = ShapeUtils::evalDimsToExclude(x->rankOf(), 1, zero.data());
     auto tadsX = x->allTensorsAlongDimension(*dims);
     auto tadsY = y->allTensorsAlongDimension(*dims);
     auto tadsZ = z->allTensorsAlongDimension(*dims);

     for (int e = 0; e < tadsX.size(); e++) {
       if (!evaluateCondition(condition, e)) {
         tadsZ.at(e)->assign(tadsY.at(e));
       } else {
         tadsZ.at(e)->assign(tadsX.at(e));
       }
     }

     delete dims;
   }
   // Case 4: Invalid shapes - provide detailed error message
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
   // in this case we return 2D matrix, which basically contains coordinates fo true
   REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead",
                block.width());
   auto output = OUTPUT_VARIABLE(0);
   std::vector<LongType> zero({0});

   int width = condition->rankOf();
   if (z->isEmpty()) return Status::OK;

   std::vector<LongType> *dims = ShapeUtils::evalDimsToExclude(width,1,zero.data());

   helpers::_where(block.launchContext(), *condition, *output, block.workspace());
   delete dims;
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
   // FIXME: we can't estimate result here in this case
   // output shape is the 2D tensor num_true x rankOf (inShape)
   auto condition = INPUT_VARIABLE(0);
   auto inShape = inputShape->at(0);
   LongType numOfTrue = 0;  // condition->reduceNumber(reduce::CountNonZero, nullptr).e<sd::LongType>(0);
   for (LongType i = 0; i < condition->lengthOf(); i++)
     if (evaluateCondition(condition, i)) numOfTrue++;

   LongType * theNewShape;
   if (numOfTrue > 0) {
     LongType* newShape;
     ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
     newShape[0] = 2;
     newShape[1] = numOfTrue;
     newShape[2] = shape::rank(inShape);
     newShape[3] = 1;
     newShape[4] = 1;
     newShape[5] = 0;
     newShape[6] = 1;
     newShape[7] = 99;
#if defined(HAS_LONG)
     ShapeUtils::updateStridesAndType(newShape, INT64, 'c');
#else
     // Fallback to INT32 if INT64 is not available
     ShapeUtils::updateStridesAndType(newShape, INT32, 'c');
#endif

     theNewShape = CONSTANT(newShape);
     RELEASE(newShape, block.getWorkspace());
   } else {
#if defined(HAS_LONG)
     theNewShape = ConstantShapeHelper::getInstance().emptyShapeInfo(INT64);
#else
     // Fallback to INT32 if INT64 is not available
     theNewShape = ConstantShapeHelper::getInstance().emptyShapeInfo(INT32);
#endif
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
}
}  // namespace ops
}  // namespace sd

#endif