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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.03.2019
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/gather.h>
#include <legacy/NativeOpExecutioner.h>

#include <numeric>
#if NOT_EXCLUDED(OP_gather)
namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
void gather(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output,
            const std::vector<LongType>& intArgs) {
  
  // Validate input parameters
  if (input == nullptr) {
    THROW_EXCEPTION("Gather operation: input array is null");
  }
  if (output == nullptr) {
    THROW_EXCEPTION("Gather operation: output array is null");
  }
  
  sd::LongType axis = intArgs.size() > 0 ? intArgs[0] : static_cast<LongType>(0);
  const sd::LongType inputRank = input->rankOf();
  
  // Normalize negative axis
  if (axis < 0) axis += inputRank;
  
  // Validate axis bounds
  if (axis >= inputRank || axis < 0) {
    THROW_EXCEPTION("Gather operation: axis is out of bounds for input array");
  }

  const sd::LongType numOfIntArgs = intArgs.size();

  if (indices != nullptr) {
    // Validate indices array
    if (indices->isEmpty() && !output->isEmpty()) {
      THROW_EXCEPTION("Gather operation: indices array is empty but output is not empty");
    }
    
    // first case: indices consist of only one scalar
    if (indices->isScalar()) {
      auto idx = indices->e<sd::LongType>(0);
      
      // Validate index bounds
      if (idx < 0 || idx >= input->sizeAt(axis)) {
        THROW_EXCEPTION("Gather operation: index is out of bounds for the specified axis");
      }
      
      if (input->rankOf() <= 1) {
        // For scalar indices, rank 0 or 1 input: use only element accessor methods
        // These should be the most basic operations that don't trigger TAD
        try {
          // Use type-specific element access and assignment
          switch (input->dataType()) {
            case FLOAT32: {
              auto value = input->e<float>(idx);
              output->p(0, value);
              break;
            }
            case DOUBLE: {
              auto value = input->e<double>(idx);
              output->p(0, value);
              break;
            }
            case INT32: {
              auto value = input->e<int32_t>(idx);
              output->p(0, value);
              break;
            }
            case INT64: {
              auto value = input->e<sd::LongType>(idx);
              output->p(0, value);
              break;
            }
            case BOOL: {
              auto value = input->e<bool>(idx);
              output->p(0, value);
              break;
            }
            case INT8: {
              auto value = input->e<int8_t>(idx);
              output->p(0, value);
              break;
            }
            case INT16: {
              auto value = input->e<int16_t>(idx);
              output->p(0, value);
              break;
            }
            case UINT8: {
              auto value = input->e<uint8_t>(idx);
              output->p(0, value);
              break;
            }
            case UINT16: {
              auto value = input->e<uint16_t>(idx);
              output->p(0, value);
              break;
            }
            case UINT32: {
              auto value = input->e<uint32_t>(idx);
              output->p(0, value);
              break;
            }
            case UINT64: {
              auto value = input->e<uint64_t>(idx);
              output->p(0, value);
              break;
            }
            case HALF: {
              auto value = input->e<float16>(idx);
              output->p(0, value);
              break;
            }
            case BFLOAT16: {
              auto value = input->e<bfloat16>(idx);
              output->p(0, value);
              break;
            }
            default: {
              // Generic fallback using double
              auto value = input->e<double>(idx);
              output->p(0, value);
              break;
            }
          }
        } catch (const std::exception& e) {
          THROW_EXCEPTION("Gather operation: failed to access element in simple case");
        }
      } else {
        // For higher rank arrays, use the slice operation
        NDArray inSubArr = (*input)(idx, {axis});
        output->assign(&inSubArr);
      }
    } else {
      // Validate all indices
      for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
        auto idx = indices->e<sd::LongType>(i);
        if (idx < 0 || idx >= input->sizeAt(axis)) {
          THROW_EXCEPTION("Gather operation: one or more indices are out of bounds for the specified axis");
        }
      }
      
      if (input->rankOf() == 1 && output->rankOf() == 1) {
        // Simple 1D to 1D case - use only element accessors
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto curr = indices->e<sd::LongType>(i);
            
            try {
              // Use type-specific element access for maximum compatibility
              switch (input->dataType()) {
                case FLOAT32: {
                  auto value = input->e<float>(curr);
                  output->p(i, value);
                  break;
                }
                case DOUBLE: {
                  auto value = input->e<double>(curr);
                  output->p(i, value);
                  break;
                }
                case INT32: {
                  auto value = input->e<int32_t>(curr);
                  output->p(i, value);
                  break;
                }
                case INT64: {
                  auto value = input->e<int64_t>(curr);
                  output->p(i, value);
                  break;
                }
                case BOOL: {
                  auto value = input->e<bool>(curr);
                  output->p(i, value);
                  break;
                }
                default: {
                  // Generic fallback
                  auto value = input->e<double>(curr);
                  output->p(i, value);
                  break;
                }
              }
            } catch (const std::exception& e) {
              // Skip this element if there's an error
              continue;
            }
          }
        };

        samediff::Threads::parallel_for(func, 0, output->lengthOf());

      } else {
        std::vector<sd::LongType> dimsOut;
        for (sd::LongType i = 0; i < axis; ++i) dimsOut.push_back(i);
        for (sd::LongType i = axis + indices->rankOf(); i < output->rankOf(); ++i) dimsOut.push_back(i);

        std::vector<sd::LongType> axesVec = {axis};
        std::vector<sd::LongType> *dimsIn = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, axesVec.data());
        
        // Check if dimensions calculation failed
        if (dimsIn == nullptr) {
          THROW_EXCEPTION("Gather operation: failed to evaluate dimensions to exclude");
        }

        const sd::LongType numOfSubArrs = indices->lengthOf();

        // Safely get TAD packs with null checking
        TadPack* inTadPack = nullptr;
        TadPack* outTadPack = nullptr;
        
        try {
          inTadPack = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimsIn);
          if (inTadPack == nullptr) {
            delete dimsIn;
            THROW_EXCEPTION("Gather operation: failed to create input TAD pack");
          }
          
          outTadPack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dimsOut);
          if (outTadPack == nullptr) {
            delete dimsIn;
            THROW_EXCEPTION("Gather operation: failed to create output TAD pack");
          }
        } catch (...) {
          delete dimsIn;
          throw;
        }
        
        delete dimsIn;
        
        auto inTadShapeInfo = inTadPack->primaryShapeInfo();
        auto outTadShapeInfo = outTadPack->primaryShapeInfo();

        if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' &&
            input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 &&
            shape::elementWiseStride(outTadShapeInfo) == 1) {
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto idx = indices->e<sd::LongType>(i);
              auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[idx]);
              auto outBuff = output->bufferWithOffset(outTadPack->primaryOffsets()[i]);

              memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
            }
          };
          samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        } else {
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto idx = indices->e<sd::LongType>(i);
              auto offset = inTadPack->primaryOffsets()[idx];
              auto inBuff = input->bufferWithOffset(offset);
              auto outOffset = outTadPack->primaryOffsets()[i];
              auto outBuff = output->bufferWithOffset(outOffset);
              NativeOpExecutioner::execTransformAny(input->getContext(), transform::Assign, inBuff, inTadShapeInfo,
                                                    nullptr /*input specialBuffer*/, nullptr /*input special*/, outBuff,
                                                    outTadShapeInfo, nullptr /*output specialBuffer*/,
                                                    nullptr /*output special*/, nullptr, false /*allowParallelism*/);
            }
          };

          samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        }
      }
    }
  } else {
    // we only allow scalar/vector case here when indices is null
    if (numOfIntArgs < 2) {
      THROW_EXCEPTION("Gather operation: indices should be provided either as array or as integer arguments");
    }
    
    if (numOfIntArgs == 2) {  // scalar case
      auto idx = intArgs[1];
      
      // Validate index bounds
      if (idx < 0 || idx >= input->sizeAt(axis)) {
        THROW_EXCEPTION("Gather operation: index argument is out of bounds for the specified axis");
      }
      
      if (input->rankOf() <= 1) {
        // For rank 0 or 1 input, use only element accessors
        try {
          switch (input->dataType()) {
            case FLOAT32: {
              auto value = input->e<float>(idx);
              output->p(0, value);
              break;
            }
            case DOUBLE: {
              auto value = input->e<double>(idx);
              output->p(0, value);
              break;
            }
            case INT32: {
              auto value = input->e<int32_t>(idx);
              output->p(0, value);
              break;
            }
            case INT64: {
              auto value = input->e<sd::LongType>(idx);
              output->p(0, value);
              break;
            }
            default: {
              auto value = input->e<double>(idx);
              output->p(0, value);
              break;
            }
          }
        } catch (const std::exception& e) {
          THROW_EXCEPTION("Gather operation: failed to access element in integer argument case");
        }
      } else {
        NDArray assign = (*input)(idx, {axis});
        output->assign(&assign);
      }
    } else {  // vector case
      const sd::LongType numOfSubArrs = intArgs.size() - 1;
      
      // Validate all index arguments
      for (sd::LongType i = 1; i < numOfIntArgs; i++) {
        auto idx = intArgs[i];
        if (idx < 0 || idx >= input->sizeAt(axis)) {
          THROW_EXCEPTION("Gather operation: one or more index arguments are out of bounds for the specified axis");
        }
      }

      std::vector<sd::LongType> axesVec = {axis};
      std::vector<sd::LongType> *dims = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, axesVec.data());
      
      // Check if dimensions calculation failed
      if (dims == nullptr) {
        THROW_EXCEPTION("Gather operation: failed to evaluate dimensions to exclude for vector case");
      }

      // Safely get TAD packs with null checking
      TadPack* inTadPack = nullptr;
      TadPack* outTadPack = nullptr;
      
      try {
        inTadPack = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dims);
        if (inTadPack == nullptr) {
          delete dims;
          THROW_EXCEPTION("Gather operation: failed to create input TAD pack for vector case");
        }
        
        outTadPack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dims);
        if (outTadPack == nullptr) {
          delete dims;
          THROW_EXCEPTION("Gather operation: failed to create output TAD pack for vector case");
        }
      } catch (...) {
        delete dims;
        throw;
      }
      
      delete dims;

      auto inTadShapeInfo = inTadPack->primaryShapeInfo();
      auto outTadShapeInfo = outTadPack->primaryShapeInfo();

      if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' &&
          input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 &&
          shape::elementWiseStride(outTadShapeInfo) == 1) {
        auto func = PRAGMA_THREADS_FOR {
          for (sd::LongType i = start; i < stop; i++) {
            auto idx = intArgs[i + 1];
            auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[idx]);
            void* outBuff = output->bufferWithOffset(outTadPack->primaryOffsets()[i]);

            std::memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
          }
        };
        samediff::Threads::parallel_tad(func, 0, numOfSubArrs);

      } else {
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto idx = intArgs[i + 1];
            auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[idx]);
            auto outBuff = output->bufferWithOffset(outTadPack->primaryOffsets()[i]);

            NativeOpExecutioner::execTransformAny(input->getContext(), transform::Assign, inBuff, inTadShapeInfo,
                                                  nullptr /*input specialBuffer*/, nullptr /*input special*/, outBuff,
                                                  outTadShapeInfo, nullptr /*output specialBuffer*/,
                                                  nullptr /*output special*/, nullptr, false /*allowParallelism*/);
          }
        };
        samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif