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
  sd::LongType axis = intArgs.size() > 0 ? intArgs[0] :static_cast<LongType>(0);
  const sd::LongType inputRank = input->rankOf();
  if (axis < 0) axis += inputRank;

  const sd::LongType numOfIntArgs = intArgs.size();

  if (indices != nullptr) {
    // first case: indices consist of only one scalar
    if (indices->isScalar()) {
      if (input->rankOf() <= 1) {
        // For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead,
        // we want to get a scalar
        auto idx = indices->e<sd::LongType>(0);
        auto scalarNDArray = input->e(idx);
        output->assign(scalarNDArray);
      } else {
        NDArray inSubArr = (*input)(indices->e<sd::LongType>(0), {axis});
        output->assign(inSubArr);
      }
    } else {
      if (input->rankOf() == 1 && output->rankOf() == 1) {
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto curr = indices->e<sd::LongType>(i);
            output->p(i, curr);
          }
        };

        samediff::Threads::parallel_for(func, 0, output->lengthOf());

      } else {
        std::vector<sd::LongType> dimsOut;
        for (sd::LongType i = 0; i < axis; ++i) dimsOut.push_back(i);
        for (sd::LongType i = axis + indices->rankOf(); i < output->rankOf(); ++i) dimsOut.push_back(i);

        std::vector<sd::LongType> axesVec = {axis};
        std::vector<sd::LongType> *dimsIn = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,axesVec.data());

        const sd::LongType numOfSubArrs = indices->lengthOf();

        auto inTadPack = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimsIn);
        delete dimsIn;
        auto outTadPack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dimsOut);
        auto inTadShapeInfo = inTadPack->primaryShapeInfo();
        auto outTadShapeInfo = outTadPack->primaryShapeInfo();

        if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' &&
            input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 &&
            shape::elementWiseStride(outTadShapeInfo) == 1) {
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[indices->e<sd::LongType>(i)]);
              auto outBuff = output->bufferWithOffset(outTadPack->primaryOffsets()[i]);

              memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
            }
          };
          samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        } else {
          auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
              auto offset = inTadPack->primaryOffsets()[indices->e<sd::LongType>(i)];
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
    // we only allow scalar/vector case here
    if (numOfIntArgs == 2) {  // scalar case
      output->assign((*input)(intArgs[1], {axis}));
    } else {  // vector case
      const sd::LongType numOfSubArrs = intArgs.size() - 1;

      std::vector<sd::LongType> axesVec = {axis};
      std::vector<sd::LongType> *dims = ShapeUtils::evalDimsToExclude(input->rankOf(),1,axesVec.data());

      auto inTadPack = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dims);
      auto outTadPack = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dims);
      delete dims;

      auto inTadShapeInfo = inTadPack->primaryShapeInfo();
      auto outTadShapeInfo = outTadPack->primaryShapeInfo();

      if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' &&
          input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 &&
          shape::elementWiseStride(outTadShapeInfo) == 1) {
        auto func = PRAGMA_THREADS_FOR {
          for (sd::LongType i = start; i < stop; i++) {
            auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[intArgs[i + 1]]);
            void* outBuff = output->bufferWithOffset(outTadPack->primaryOffsets()[i]);

            std::memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
          }
        };
        samediff::Threads::parallel_tad(func, 0, numOfSubArrs);

      } else {
        auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto inBuff = input->bufferWithOffset(inTadPack->primaryOffsets()[intArgs[i + 1]]);
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