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
//  @author sgazeos@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/roll.h>

#include <memory>

#if NOT_EXCLUDED(OP_roll)
namespace sd {
namespace ops {
namespace helpers {

static LongType normalizedShift(LongType shift, LongType dimension) {
  if (dimension <= 1) return 0;
  shift %= dimension;
  return shift < 0 ? shift + dimension : shift;
}

template <typename T>
static void snapshotCOrder(NDArray* input, T* snapshot) {
  const auto length = input->lengthOf();
  const auto rank = input->rankOf();
  const auto shape = input->shapeOf();
  const auto strides = input->stridesOf();
  const auto buffer = input->bufferAsT<T>();

  auto snapshotLoop = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; ++e) {
      LongType coords[SD_MAX_RANK];
      LongType offset;
      INDEX2COORDS(e, rank, shape, coords);
      COORDS2INDEX(rank, strides, coords, offset);
      snapshot[static_cast<size_t>(e)] = buffer[offset];
    }
  };
  samediff::Threads::parallel_for(snapshotLoop, 0, length);
}

template <typename T>
static void rollFunctorLinear_(NDArray* input, NDArray* output, LongType shift, bool inplace) {
  const auto length = input->lengthOf();
  if (length == 0) return;

  NDArray::preparePrimaryUse({output}, {input});
  std::unique_ptr<T[]> snapshot;
  if (inplace) {
    snapshot.reset(new T[static_cast<size_t>(length)]);
    snapshotCOrder(input, snapshot.get());
  }

  const auto actualShift = normalizedShift(shift, length);
  const auto inputRank = input->rankOf();
  const auto inputShape = input->shapeOf();
  const auto inputStrides = input->stridesOf();
  const auto inputBuffer = input->bufferAsT<T>();
  const auto outputRank = output->rankOf();
  const auto outputShape = output->shapeOf();
  const auto outputStrides = output->stridesOf();
  auto outputBuffer = output->bufferAsT<T>();

  auto rollLoop = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; ++e) {
      const auto sourceIndex = (e + length - actualShift) % length;
      T value;
      if (inplace) {
        value = snapshot[static_cast<size_t>(sourceIndex)];
      } else {
        LongType inputCoords[SD_MAX_RANK];
        LongType inputOffset;
        INDEX2COORDS(sourceIndex, inputRank, inputShape, inputCoords);
        COORDS2INDEX(inputRank, inputStrides, inputCoords, inputOffset);
        value = inputBuffer[inputOffset];
      }

      LongType outputCoords[SD_MAX_RANK];
      LongType outputOffset;
      INDEX2COORDS(e, outputRank, outputShape, outputCoords);
      COORDS2INDEX(outputRank, outputStrides, outputCoords, outputOffset);
      outputBuffer[outputOffset] = value;
    }
  };
  samediff::Threads::parallel_for(rollLoop, 0, length);
  NDArray::registerPrimaryUse({output}, {input});
}

template <typename T>
static void rollFunctorFull_(NDArray* input, NDArray* output, const std::vector<LongType>& shifts,
                             const std::vector<LongType>& axes, bool inplace) {
  const auto length = input->lengthOf();
  if (length == 0) return;

  NDArray::preparePrimaryUse({output}, {input});
  std::unique_ptr<T[]> snapshot;
  if (inplace) {
    snapshot.reset(new T[static_cast<size_t>(length)]);
    snapshotCOrder(input, snapshot.get());
  }

  const auto rank = output->rankOf();
  const auto shape = output->shapeOf();
  const auto inputStrides = input->stridesOf();
  const auto inputBuffer = input->bufferAsT<T>();
  const auto outputStrides = output->stridesOf();
  auto outputBuffer = output->bufferAsT<T>();

  auto rollLoop = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; ++e) {
      LongType sourceCoords[SD_MAX_RANK];
      INDEX2COORDS(e, rank, shape, sourceCoords);

      for (size_t i = 0; i < axes.size(); ++i) {
        const auto axis = axes[i];
        const auto dimension = shape[axis];
        const auto shift = normalizedShift(shifts[i], dimension);
        sourceCoords[axis] = (sourceCoords[axis] + dimension - shift) % dimension;
      }

      T value;
      if (inplace) {
        LongType sourceIndex = 0;
        for (LongType dimension = 0; dimension < rank; ++dimension) {
          sourceIndex = sourceIndex * shape[dimension] + sourceCoords[dimension];
        }
        value = snapshot[static_cast<size_t>(sourceIndex)];
      } else {
        LongType inputOffset;
        COORDS2INDEX(rank, inputStrides, sourceCoords, inputOffset);
        value = inputBuffer[inputOffset];
      }

      LongType outputCoords[SD_MAX_RANK];
      LongType outputOffset;
      INDEX2COORDS(e, rank, shape, outputCoords);
      COORDS2INDEX(rank, outputStrides, outputCoords, outputOffset);
      outputBuffer[outputOffset] = value;
    }
  };
  samediff::Threads::parallel_for(rollLoop, 0, length);
  NDArray::registerPrimaryUse({output}, {input});
}

void rollFunctorFull(sd::LaunchContext* context, NDArray* input, NDArray* output,
                     const std::vector<LongType>& shifts, const std::vector<LongType>& axes, bool inplace) {
  BUILD_SINGLE_SELECTOR(input->dataType(), rollFunctorFull_, (input, output, shifts, axes, inplace), SD_COMMON_TYPES);
}

void rollFunctorLinear(sd::LaunchContext* context, NDArray* input, NDArray* output, LongType shift, bool inplace) {
  BUILD_SINGLE_SELECTOR(input->dataType(), rollFunctorLinear_, (input, output, shift, inplace), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(void rollFunctorLinear_, (NDArray* input, NDArray* output, LongType shift, bool inplace),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(void rollFunctorFull_,
                      (NDArray* input, NDArray* output, const std::vector<LongType>& shifts,
                       const std::vector<LongType>& axes, bool inplace),
                      SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif