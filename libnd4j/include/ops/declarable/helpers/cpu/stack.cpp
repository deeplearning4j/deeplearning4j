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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/ResultSet.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/stack.h>

#include <legacy/NativeOpExecutioner.h>
#if NOT_EXCLUDED(OP_stack)
namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
static void stack_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int dim) {
  const int numOfSubArrs = inArrs.size();

  //no op on empty
  if (inArrs[0]->rankOf() == 0 && !inArrs[0]->isEmpty()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) if(!output.isEmpty() && !inArrs[i]->isEmpty()) output.p<T>(i, inArrs[i]->t<T>(0));
    };

    samediff::Threads::parallel_for(func, 0, numOfSubArrs);
  } else if(!output.isEmpty()) {
    std::vector<sd::LongType> dimVec = {dim};
    auto vec = ShapeUtils::evalDimsToExclude(output.rankOf(),1,dimVec.data());
    auto zTadPack = ConstantTadHelper::getInstance().tadForDimensions(
        output.shapeInfo(), vec);
    auto zTadShapeInfo = zTadPack->primaryShapeInfo();
    delete vec;
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        void* zBuff = output.bufferWithOffset(zTadPack->primaryOffsets()[i]);

        NativeOpExecutioner::execTransformAny(
            inArrs[0]->getContext(), transform::Assign, inArrs[i]->buffer(), inArrs[i]->shapeInfo(),
            nullptr /*input specialBuffer*/, nullptr /*input special*/, zBuff, zTadShapeInfo,
            nullptr /*output specialBuffer*/, nullptr /*output special*/, nullptr, false /*allowParallelism*/);
      }
    };

    samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
  }
}

////////////////////////////////////////////////////////////////////////
void stack(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int dim) {
  BUILD_SINGLE_SELECTOR(output.dataType(), stack_, (context, inArrs, output, dim), SD_COMMON_TYPES);
}
BUILD_SINGLE_TEMPLATE( void stack_, 
                      (LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int dim),
                      SD_COMMON_TYPES);

///////////////////////////////////////////////////////////////////
template <typename T>
static void unstack_(LaunchContext* context, NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {
  const int numOfSubArrs = outArrs.size();

  if (outArrs[0]->rankOf() == 0) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) outArrs[i]->p<T>(0, input.t<T>(i));
    };

    samediff::Threads::parallel_for(func, 0, numOfSubArrs);
  } else {
    std::vector<sd::LongType> dimVec = {dim};
    auto vec = ShapeUtils::evalDimsToExclude(input.rankOf(), 1,dimVec.data());
    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(
        input.shapeInfo(), vec);
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();
    delete vec;
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto xBuff = input.bufferWithOffset(xTadPack->primaryOffsets()[i]);

        NativeOpExecutioner::execTransformAny(
            input.getContext(), transform::Assign, xBuff, xTadShapeInfo, nullptr /*input specialBuffer*/,
            nullptr /*input special*/, outArrs[i]->buffer(), outArrs[i]->shapeInfo(), nullptr /*output specialBuffer*/,
            nullptr /*output special*/, nullptr, false /*allowParallelism*/);
      }
    };

    samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
  }
}

////////////////////////////////////////////////////////////////////////
void unstack(LaunchContext* context, NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {
  BUILD_SINGLE_SELECTOR(input.dataType(), unstack_, (context, input, outArrs, dim), SD_COMMON_TYPES);
}
BUILD_SINGLE_TEMPLATE( void unstack_,
                      (LaunchContext* context, NDArray& input, const std::vector<NDArray*>& outArrs, const int dim), 
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif