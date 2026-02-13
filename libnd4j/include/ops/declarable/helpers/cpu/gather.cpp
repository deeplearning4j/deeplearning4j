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
#include <cstring>

#include <numeric>

// Helper to check if TAD is contiguous for memcpy optimization
static bool isTadContiguous(const sd::LongType* tadShapeInfo) {
  const int rank = shape::rank(tadShapeInfo);
  if (rank == 0) return true;

  const sd::LongType* shapePtr = shape::shapeOf(tadShapeInfo);
  const sd::LongType* stridePtr = shape::stride(tadShapeInfo);

  sd::LongType expectedStride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    if (shapePtr[i] == 1) continue;
    if (stridePtr[i] != expectedStride) return false;
    expectedStride *= shapePtr[i];
  }
  return true;
}
#if NOT_EXCLUDED(OP_gather)
namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
void gather(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output,
            const std::vector<LongType>& intArgs) {
  sd::LongType axis = intArgs.size() > 0 ? intArgs[0] : 0;
  const sd::LongType inputRank = input->rankOf();
  if (axis < 0) axis += inputRank;

  const sd::LongType numOfIntArgs = intArgs.size();

  // Special handling for 1D input with axis=0
  // This handles cases like gathering from shape arrays where we want flat indexing
  bool is1DFlatGather = (inputRank == 1 && axis == 0);

  if (indices != nullptr) {
    // NOTE: Index validation is done in the main gather op when checkIndices=true
    // We skip redundant validation here for performance

    // Pre-fetch all indices to avoid virtual function calls in hot loop
    const sd::LongType numIndices = indices->lengthOf();
    const sd::LongType inputLen = input->lengthOf();
    std::vector<sd::LongType> indicesVec(numIndices);
    for (sd::LongType i = 0; i < numIndices; i++) {
      sd::LongType idx = indices->e<sd::LongType>(i);
      // Clamp to valid range [0, inputLen-1] to prevent OOB
      if (idx < 0) idx = 0;
      if (idx >= inputLen) idx = inputLen - 1;
      indicesVec[i] = idx;
    }

    if (is1DFlatGather) {
      // Special case: 1D input with axis=0 - treat as flat array gather
      // Use direct buffer access for better performance
      const auto inputType = input->dataType();
      const auto outputType = output->dataType();

      // Fast path for common type combinations
      if (inputType == outputType && inputType == sd::DataType::FLOAT32) {
        auto inBuff = input->bufferAsT<float>();
        auto outBuff = output->bufferAsT<float>();
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            outBuff[i] = inBuff[indicesVec[i]];
          }
        };
        samediff::Threads::parallel_for(func, 0, numIndices);
      } else if (inputType == outputType && inputType == sd::DataType::DOUBLE) {
        auto inBuff = input->bufferAsT<double>();
        auto outBuff = output->bufferAsT<double>();
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            outBuff[i] = inBuff[indicesVec[i]];
          }
        };
        samediff::Threads::parallel_for(func, 0, numIndices);
      } else if (inputType == outputType && inputType == sd::DataType::INT64) {
        auto inBuff = input->bufferAsT<sd::LongType>();
        auto outBuff = output->bufferAsT<sd::LongType>();
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            outBuff[i] = inBuff[indicesVec[i]];
          }
        };
        samediff::Threads::parallel_for(func, 0, numIndices);
      } else {
        // Fallback for other types
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto value = input->e<double>(indicesVec[i]);
            output->p(i, value);
          }
        };
        samediff::Threads::parallel_for(func, 0, numIndices);
      }

    } else {
      // Standard gather implementation
      //
      // For gather with axis=A on input shape [..., dimA, ...] and indices shape [I1, I2, ...]:
      // - Output shape is: input[0:A] + indices_shape + input[A+1:]
      // - Input TADs: iterate along axis A, each TAD has shape input[A+1:]
      // - Output TADs: iterate along indices dimensions, each TAD has same shape as input TAD
      //
      // tadForDimensions takes dimensions to KEEP in each TAD (not to exclude)
      // It then internally calls evalDimsToExclude to find which dims to iterate over

      std::vector<sd::LongType> axesVec = {axis};
      auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, axesVec.data());

      // For output TADs, we want the same shape as input TADs
      // Input TAD shape = all dims except axis
      // Output shape = input[0:axis] + indices_shape + input[axis+1:]
      // Output TAD dims should be: dims 0 to axis-1, then dims axis+indicesRank to end
      // This gives TAD shape matching input's TAD shape
      const sd::LongType indicesRank = indices->rankOf();
      const sd::LongType outputRank = output->rankOf();

      std::vector<sd::LongType> outputTadDims;
      outputTadDims.reserve(outputRank - indicesRank);

      // Add dimensions before the indices dimensions (0 to axis-1)
      for (sd::LongType d = 0; d < axis; d++) {
        outputTadDims.push_back(d);
      }
      // Add dimensions after the indices dimensions (axis+indicesRank to outputRank-1)
      for (sd::LongType d = axis + indicesRank; d < outputRank; d++) {
        outputTadDims.push_back(d);
      }

      // If outputTadDims is empty, it means each TAD is a scalar - handle this case
      // by using the same approach as input (which would also have empty TAD dims)

      // Get TAD packs - these are cached and should not be deleted
      auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
      auto tadPackOut = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &outputTadDims);

      // Validate TAD packs before use
      if (tadPack == nullptr || tadPackOut == nullptr) {
        if (dimensions) delete dimensions;
        THROW_EXCEPTION("gather: Failed to create TAD packs");
      }

      // Now safe to delete dimensions as TAD helper has made internal copy
      delete dimensions;

      auto tadShapeInfo = tadPack->primaryShapeInfo();
      auto tadOffsets = tadPack->primaryOffsets();
      auto tadShapeInfoOut = tadPackOut->primaryShapeInfo();
      auto tadOffsetsOut = tadPackOut->primaryOffsets();

      // Validate that input and output TAD shapes match
      auto inputTadLength = shape::length(tadShapeInfo);
      auto outputTadLength = shape::length(tadShapeInfoOut);
      if (inputTadLength != outputTadLength) {
        std::string error = "gather: TAD shape mismatch - input TAD length ";
        error += std::to_string(inputTadLength);
        error += " != output TAD length ";
        error += std::to_string(outputTadLength);
        error += ". Input shape: ";
        error += ShapeUtils::shapeAsString(input->shapeInfo());
        error += ", Output shape: ";
        error += ShapeUtils::shapeAsString(output->shapeInfo());
        error += ", Indices shape: ";
        error += ShapeUtils::shapeAsString(indices->shapeInfo());
        error += ", axis: ";
        error += std::to_string(axis);
        THROW_EXCEPTION(error.c_str());
      }

      auto tadShapeInfoCast = const_cast<sd::LongType *>(tadShapeInfo);
      auto tadShapeInfoOutCast = const_cast<sd::LongType *>(tadShapeInfoOut);

      // Calculate the number of gather operations (equal to indices length)
      const sd::LongType numGatherOps = indices->lengthOf();

      // Validate bounds before parallel execution
      if (numGatherOps > tadPackOut->numberOfTads()) {
        std::string error = "gather: indices length ";
        error += std::to_string(numGatherOps);
        error += " exceeds output TAD count ";
        error += std::to_string(tadPackOut->numberOfTads());
        THROW_EXCEPTION(error.c_str());
      }

      auto numInputTads = tadPack->numberOfTads();

      // Check if we can use memcpy (contiguous TADs with same type)
      bool canUseMemcpy = isTadContiguous(tadShapeInfo) &&
                          isTadContiguous(tadShapeInfoOut) &&
                          input->dataType() == output->dataType();
      auto tadLength = shape::length(tadShapeInfo);
      auto bytesToCopy = tadLength * input->sizeOfT();

      // Cache buffer pointers for faster access in loop
      auto inputBuffer = input->buffer();
      auto outputBuffer = output->buffer();
      auto elementSize = input->sizeOfT();

      if (canUseMemcpy) {
        // Fast path: use memcpy for contiguous TADs
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto idx = indicesVec[i];
            if (idx >= numInputTads || idx < 0) continue;

            auto offsetIn = tadOffsets[idx];
            auto offsetOut = tadOffsetsOut[i];
            std::memcpy(reinterpret_cast<int8_t*>(outputBuffer) + offsetOut * elementSize,
                       reinterpret_cast<const int8_t*>(inputBuffer) + offsetIn * elementSize,
                       bytesToCopy);
          }
        };
        samediff::Threads::parallel_tad(func, 0, numGatherOps);
      } else {
        // Fallback: use NativeOpExecutioner
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto idx = indicesVec[i];
            if (idx >= numInputTads || idx < 0) continue;

            auto offsetIn = tadOffsets[idx];
            auto offsetOut = tadOffsetsOut[i];

            NativeOpExecutioner::execTransformAny(input->getContext(),
                                                  transform::Assign,
                                                  input->bufferWithOffset(offsetIn), tadShapeInfoCast,
                                                  nullptr, nullptr,
                                                  output->bufferWithOffset(offsetOut), tadShapeInfoOutCast,
                                                  nullptr, nullptr,
                                                  nullptr, false);
          }
        };
        samediff::Threads::parallel_tad(func, 0, numGatherOps);
      }
    }

  } else {
    // Integer arguments case
    // NOTE: Index validation is done in the main gather op when checkIndices=true
    // We skip redundant validation here for performance

    if (numOfIntArgs == 2) {
      if (is1DFlatGather) {
        // For 1D flat gather with single index
        auto value = input->e<double>(intArgs[1]);
        output->assign(value);
      } else {
        // Standard single index gather
        NDArray *copy = (*input)(intArgs[1], {axis});
        output->assign(copy);
        delete copy;
      }
    } else {
      if (is1DFlatGather) {
        // Multiple indices for 1D flat gather
        const sd::LongType inputLen2 = input->lengthOf();
        for (int i = 1; i < numOfIntArgs; ++i) {
          auto idx = intArgs[i];
          // Clamp to valid range to prevent OOB
          if (idx < 0) idx = 0;
          if (idx >= inputLen2) idx = inputLen2 - 1;
          auto value = input->e<double>(idx);
          output->p(i - 1, value);
        }
      } else {
        // Standard multiple indices gather
        // Use the same dimension calculation for input and output TADs
        std::vector<sd::LongType> axesVec = {axis};
        auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, axesVec.data());

        // Get TAD packs - these are cached and should not be deleted
        auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
        auto tadPackOut = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);

        // Validate TAD packs before use
        if (tadPack == nullptr || tadPackOut == nullptr) {
          if (dimensions) delete dimensions;
          THROW_EXCEPTION("gather: Failed to create TAD packs");
        }

        // Now safe to delete dimensions as TAD helper has made internal copy
        delete dimensions;

        auto tadShapeInfo = tadPack->primaryShapeInfo();
        auto tadOffsets = tadPack->primaryOffsets();
        auto tadShapeInfoOut = tadPackOut->primaryShapeInfo();
        auto tadOffsetsOut = tadPackOut->primaryOffsets();

        // Validate that input and output TAD shapes match
        auto inputTadLength = shape::length(tadShapeInfo);
        auto outputTadLength = shape::length(tadShapeInfoOut);
        if (inputTadLength != outputTadLength) {
          std::string error = "gather: TAD shape mismatch - input TAD length ";
          error += std::to_string(inputTadLength);
          error += " != output TAD length ";
          error += std::to_string(outputTadLength);
          error += ". Input shape: ";
          error += ShapeUtils::shapeAsString(input->shapeInfo());
          error += ", Output shape: ";
          error += ShapeUtils::shapeAsString(output->shapeInfo());
          error += ", axis: ";
          error += std::to_string(axis);
          THROW_EXCEPTION(error.c_str());
        }

        // Number of gather operations (number of indices provided as int args)
        const sd::LongType numGatherOps = numOfIntArgs - 1;

        // Validate bounds before parallel execution
        if (numGatherOps > tadPackOut->numberOfTads()) {
          std::string error = "gather: number of indices ";
          error += std::to_string(numGatherOps);
          error += " exceeds output TAD count ";
          error += std::to_string(tadPackOut->numberOfTads());
          THROW_EXCEPTION(error.c_str());
        }

        auto numInputTads = tadPack->numberOfTads();

        // Check if we can use memcpy (contiguous TADs with same type)
        bool canUseMemcpy = isTadContiguous(tadShapeInfo) &&
                            isTadContiguous(tadShapeInfoOut) &&
                            input->dataType() == output->dataType();
        auto tadLength = shape::length(tadShapeInfo);
        auto bytesToCopy = tadLength * input->sizeOfT();

        if (canUseMemcpy) {
          // Fast path: use memcpy for contiguous TADs
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto idx = intArgs[i + 1];
              if (idx >= numInputTads || idx < 0) continue;

              auto offsetIn = tadOffsets[idx];
              auto offsetOut = tadOffsetsOut[i];
              std::memcpy(output->bufferWithOffset(offsetOut),
                         input->bufferWithOffset(offsetIn),
                         bytesToCopy);
            }
          };
          samediff::Threads::parallel_tad(func, 0, numGatherOps);
        } else {
          // Fallback: use NativeOpExecutioner
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto idx = intArgs[i + 1];
              if (idx >= numInputTads || idx < 0) continue;

              auto offsetIn = tadOffsets[idx];
              auto offsetOut = tadOffsetsOut[i];

              NativeOpExecutioner::execTransformAny(input->getContext(),
                                                    transform::Assign,
                                                    input->bufferWithOffset(offsetIn), const_cast<sd::LongType*>(tadShapeInfo),
                                                    nullptr, nullptr,
                                                    output->bufferWithOffset(offsetOut), const_cast<sd::LongType*>(tadShapeInfoOut),
                                                    nullptr, nullptr,
                                                    nullptr, false);
            }
          };
          samediff::Threads::parallel_tad(func, 0, numGatherOps);
        }
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
