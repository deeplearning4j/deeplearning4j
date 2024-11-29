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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <helpers/Loops.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void gatherND_(NDArray& input, NDArray& indices, NDArray& output) {
  const X* x = reinterpret_cast<X*>(input.buffer());
  const Y* y = reinterpret_cast<Y*>(indices.buffer());
  X* z = reinterpret_cast<X*>(output.buffer());

  const sd::LongType xRank = input.rankOf();
  const sd::LongType yRank = indices.rankOf();
  const sd::LongType zRank = output.rankOf();
  const sd::LongType maxRank = sd::math::sd_max<sd::LongType>(yRank, sd::math::sd_max<sd::LongType>(xRank, zRank));

  const sd::LongType zLen = output.lengthOf();

  const sd::LongType yLastDim = indices.sizeAt(-1);

  const int diff = zRank - xRank;
  const bool bEqual = yLastDim == xRank;

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType xCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, output.rankOf(), output.shapeInfo(), zCoords);

      sd::LongType zOffset;
      COORDS2INDEX(output.rankOf(), shape::shapeOf(output.shapeInfo()), zCoords, zOffset);

      temp = zCoords[yRank - 1];
      zCoords[yRank - 1] = 0;

      sd::LongType yOffset;
      COORDS2INDEX(indices.rankOf(), shape::shapeOf(indices.shapeInfo()), zCoords, yOffset);

      zCoords[yRank - 1] = temp;

      if (bEqual)
        memcpy(xCoords, zCoords, zRank * sizeof(sd::LongType));
      else if (diff >= 0)
        memcpy(xCoords, zCoords + diff, xRank * sizeof(sd::LongType));
      else
        memcpy(xCoords - diff, zCoords, zRank * sizeof(sd::LongType));

      for (sd::LongType j = 0; j < yLastDim; ++j)
        xCoords[j] = y[yOffset + j * indices.stridesOf()[yRank - 1]];  // last stride

      sd::LongType xOffset;
      COORDS2INDEX(input.rankOf(), shape::shapeOf(input.shapeInfo()), xCoords, xOffset);

      z[zOffset] = x[xOffset];
    }
  };

  samediff::Threads::parallel_tad(func, 0, zLen);
}

////////////////////////////////////////////////////////////////////////
void gatherND(sd::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output) {
  BUILD_DOUBLE_SELECTOR(input.dataType(), indices.dataType(), gatherND_, (input, indices, output), SD_COMMON_TYPES,
                        SD_INDEXING_TYPES);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void gather_(NDArray* input, NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
  int axis = intArgs.size() > 0 ? intArgs[0] : 0;
  const int inputRank = input->rankOf();
  if (axis < 0) axis += inputRank;

  const int numOfIntArgs = intArgs.size();

  if (indices != nullptr) {
    for (sd::LongType i = 0; i < indices->lengthOf(); ++i)
      if (indices->e<sd::LongType>(i) >= input->sizeAt(axis))
        THROW_EXCEPTION(
            "helpers::gather function: indices array contains wrong elements, each element must be smaller than "
            "corresponding dimension of input array !");

    // first case: indices consist of only one scalar
    if (indices->isScalar()) {
      if (input->rankOf() <= 1) {
        // For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead,
        // we want to get a scalar
        auto idx = indices->e<sd::LongType>(0);
        auto scalarNDArray = input->e(idx);
        output->assign(scalarNDArray);
      } else {
        std::vector<sd::LongType> axesVec = {axis};
        auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,axesVec.data());
        auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);

        auto tadArr = NDArray(reinterpret_cast<void*>(reinterpret_cast<T*>(input->buffer()) +
                                                      tadPack->primaryOffsets()[indices->e<sd::LongType>(0)]),
                              tadPack->primaryShapeInfo(), output->getContext(), 0, 0);
        output->assign(tadArr);
        delete dimensions;

      }
    } else if (input->rankOf() == 1 && indices->isVector()) {
      // special case
      auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) output->p(e, input->e<T>(indices->e<sd::LongType>(e)));
      };

      samediff::Threads::parallel_for(func, 0, indices->lengthOf());
    } else {
      std::vector<sd::LongType> dimsOut(indices->rankOf());
      std::iota(dimsOut.begin(), dimsOut.end(), axis);  // fill with axis, axis+1, ... indices->rankOf()-1
      const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->shapeInfo(), dimsOut);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          NDArray subArrOut = (*output)(i, dimsOut);
          NDArray subArrIn = (*input)(indices->e<sd::LongType>(i), {axis});
          subArrOut.assign(subArrIn);
        }
      };

      samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
    }
  } else {
    for (int i = 1; i < numOfIntArgs; ++i)
      if (intArgs[i] >= input->sizeAt(axis))
        THROW_EXCEPTION(
            "helpers::gather function: some of input indexes is larger than corresponding shape of input array !");

    // we only allow scalar/vector case here
    if (numOfIntArgs == 2) {  // scalar case
      NDArray view = (*input)(intArgs[1], {axis});
      output->assign(view);
    } else {  // vector case
      const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->shapeInfo(), {axis});

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          NDArray subArrOut = (*output)(i, {axis});
          NDArray subArrIn = (*input)(intArgs[i + 1], {axis});
          subArrOut.assign(subArrIn);
        }
      };

      samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
    }
  }
}

void gather(NDArray* input, NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
  BUILD_SINGLE_SELECTOR(input->dataType(), gather_, (input, indices, output, intArgs), SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
