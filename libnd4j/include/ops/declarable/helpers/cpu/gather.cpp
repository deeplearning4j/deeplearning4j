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
  sd::LongType axis = intArgs.size() > 0 ? intArgs[0] : 0;
  const sd::LongType inputRank = input->rankOf();
  if (axis < 0) axis += inputRank;

  const sd::LongType numOfIntArgs = intArgs.size();

  // Special handling for 1D input with axis=0
  // This handles cases like gathering from shape arrays where we want flat indexing
  bool is1DFlatGather = (inputRank == 1 && axis == 0);

  if (indices != nullptr) {
    // Validate indices
    for (sd::LongType i = 0; i < indices->lengthOf(); ++i) {
      auto idx = indices->e<sd::LongType>(i);
      
      if (is1DFlatGather) {
        // For 1D arrays with axis=0, treat as flat array access
        if (idx >= input->lengthOf() || idx < 0) {
          std::string error = "helpers::gather function: invalid flat index ";
          error += std::to_string(idx);
          error += " at position ";
          error += std::to_string(i);
          error += ". Input is 1D with length ";
          error += std::to_string(input->lengthOf());
          error += ", valid range is [0, ";
          error += std::to_string(input->lengthOf() - 1);
          error += "]";
          THROW_EXCEPTION(error.c_str());
        }
      } else {
        // Standard axis-based validation
        if (idx >= input->sizeAt(axis) || idx < 0) {
          std::string error = "helpers::gather function: invalid index ";
          error += std::to_string(idx);
          error += " at position ";
          error += std::to_string(i);
          error += ". Input shape ";
          error += ShapeUtils::shapeAsString(input->shapeInfo());
          error += ", axis ";
          error += std::to_string(axis);
          error += ", valid range is [0, ";
          error += std::to_string(input->sizeAt(axis) - 1);
          error += "]";
          THROW_EXCEPTION(error.c_str());
        }
      }
    }

    if (is1DFlatGather) {
      // Special case: 1D input with axis=0 - treat as flat array gather
      // This handles gathering from shape arrays like [1, 512] -> gather index 1 -> get 512
      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          auto idx = indices->e<sd::LongType>(i);
          auto value = input->e<double>(idx);  // Get value at flat index
          output->p(i, value);  // Put in output at position i
        }
      };
      samediff::Threads::parallel_for(func, 0, indices->lengthOf());
      
    } else {
      // Standard gather implementation
      std::vector<sd::LongType> dimsOut(indices->rankOf());
      std::iota(dimsOut.begin(), dimsOut.end(), axis);

      const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->shapeInfo(), dimsOut);

      std::vector<sd::LongType> axesVec = {axis};
      auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, axesVec.data());
      
      // Get TAD packs - these are cached and should not be deleted
      auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
      auto tadPackOut = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dimsOut);
      
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

      auto tadShapeInfoCast = const_cast<sd::LongType *>(tadShapeInfo);
      auto tadShapeInfoOutCast = const_cast<sd::LongType *>(tadShapeInfoOut);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          auto idx = indices->e<sd::LongType>(i);
          
          if (idx >= tadPack->numberOfTads() || idx < 0) {
            continue;
          }
          
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
      samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
    }
      
  } else {
    // Integer arguments case
    for (int i = 1; i < numOfIntArgs; ++i) {
      if (is1DFlatGather) {
        // For 1D arrays with axis=0, validate against total length
        if (intArgs[i] >= input->lengthOf() || intArgs[i] < 0) {
          std::string error = "helpers::gather function: invalid flat index ";
          error += std::to_string(intArgs[i]);
          error += " at position ";
          error += std::to_string(i-1);
          error += ". Input is 1D with length ";
          error += std::to_string(input->lengthOf());
          error += ", valid range is [0, ";
          error += std::to_string(input->lengthOf() - 1);
          error += "]";
          THROW_EXCEPTION(error.c_str());
        }
      } else {
        // Standard validation
        if (intArgs[i] >= input->sizeAt(axis) || intArgs[i] < 0) {
          std::string error = "helpers::gather function: invalid index ";
          error += std::to_string(intArgs[i]);
          error += " at position ";
          error += std::to_string(i-1);
          error += ". Input shape ";
          error += ShapeUtils::shapeAsString(input->shapeInfo());
          error += ", axis ";
          error += std::to_string(axis);
          error += ", valid range is [0, ";
          error += std::to_string(input->sizeAt(axis) - 1);
          error += "]";
          THROW_EXCEPTION(error.c_str());
        }
      }
    }

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
        for (int i = 1; i < numOfIntArgs; ++i) {
          auto idx = intArgs[i];
          auto value = input->e<double>(idx);
          output->p(i - 1, value);
        }
      } else {
        // Standard multiple indices gather
        const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->shapeInfo(), {axis});

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
        
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto idx = intArgs[i + 1];
            
            if (idx >= tadPack->numberOfTads() || idx < 0) {
              continue;
            }
            
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
        samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
