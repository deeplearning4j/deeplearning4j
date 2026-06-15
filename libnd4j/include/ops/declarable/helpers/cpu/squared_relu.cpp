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
// Squared ReLU CPU implementation
// Forward:  out = max(0, x)^2
// Backward: dIn = dOut * 2 * x * (x > 0)
//

#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/squared_relu.h>

#if NOT_EXCLUDED(OP_squared_relu)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void squaredRelu_(NDArray* input, NDArray* output) {
  const auto length = input->lengthOf();
  const T* x = input->bufferAsT<T>();
  T* z = output->bufferAsT<T>();

  const auto xRank = input->rankOf();
  const auto zRank = output->rankOf();
  const auto xShape = input->shapeOf();
  const auto zShape = output->shapeOf();
  const auto xStrides = input->stridesOf();
  const auto zStrides = output->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; ++i) {
      sd::LongType xCoords[SD_MAX_RANK] = {};
      sd::LongType zCoords[SD_MAX_RANK] = {};
      sd::LongType xOffset = 0;
      sd::LongType zOffset = 0;

      INDEX2COORDS(i, xRank, xShape, xCoords);
      COORDS2INDEX(xRank, xStrides, xCoords, xOffset);

      INDEX2COORDS(i, zRank, zShape, zCoords);
      COORDS2INDEX(zRank, zStrides, zCoords, zOffset);

      T val = x[xOffset];
      T relu = val > static_cast<T>(0) ? val : static_cast<T>(0);
      z[zOffset] = relu * relu;
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void squaredRelu(LaunchContext* context, NDArray* input, NDArray* output) {
  NDArray::preparePrimaryUse({output}, {input});

  BUILD_SINGLE_SELECTOR(input->dataType(), squaredRelu_, (input, output), SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({output}, {input});
}

template <typename T>
static void squaredReluBp_(NDArray* input, NDArray* dLdO, NDArray* dLdI) {
  const auto length = input->lengthOf();
  const T* x = input->bufferAsT<T>();
  const T* gradOut = dLdO->bufferAsT<T>();
  T* gradIn = dLdI->bufferAsT<T>();

  const auto xRank = input->rankOf();
  const auto goRank = dLdO->rankOf();
  const auto giRank = dLdI->rankOf();
  const auto xShape = input->shapeOf();
  const auto goShape = dLdO->shapeOf();
  const auto giShape = dLdI->shapeOf();
  const auto xStrides = input->stridesOf();
  const auto goStrides = dLdO->stridesOf();
  const auto giStrides = dLdI->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; ++i) {
      sd::LongType xCoords[SD_MAX_RANK] = {};
      sd::LongType goCoords[SD_MAX_RANK] = {};
      sd::LongType giCoords[SD_MAX_RANK] = {};
      sd::LongType xOffset = 0;
      sd::LongType goOffset = 0;
      sd::LongType giOffset = 0;

      INDEX2COORDS(i, xRank, xShape, xCoords);
      COORDS2INDEX(xRank, xStrides, xCoords, xOffset);

      INDEX2COORDS(i, goRank, goShape, goCoords);
      COORDS2INDEX(goRank, goStrides, goCoords, goOffset);

      INDEX2COORDS(i, giRank, giShape, giCoords);
      COORDS2INDEX(giRank, giStrides, giCoords, giOffset);

      T val = x[xOffset];
      // dIn = dOut * 2 * x * (x > 0)
      gradIn[giOffset] = val > static_cast<T>(0)
                              ? gradOut[goOffset] * static_cast<T>(2) * val
                              : static_cast<T>(0);
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void squaredReluBp(LaunchContext* context, NDArray* input, NDArray* dLdO, NDArray* dLdI) {
  NDArray::preparePrimaryUse({dLdI}, {input, dLdO});

  BUILD_SINGLE_SELECTOR(input->dataType(), squaredReluBp_, (input, dLdO, dLdI), SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({dLdI}, {input, dLdO});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
