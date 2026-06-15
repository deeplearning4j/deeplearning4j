/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/lgamma.h>
#include <math/templatemath.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/PointersManager.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void lgammaKernel(const void *vx, const sd::LongType *xShapeInfo,
                                    void *vz, const sd::LongType *zShapeInfo,
                                    sd::LongType length) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset;
    sd::LongType zOffset;

    auto xRank = shape::rank(xShapeInfo);
    auto zRank = shape::rank(zShapeInfo);

    INDEX2COORDS(i, xRank, shape::shapeOf(xShapeInfo), xCoords);
    COORDS2INDEX(xRank, shape::stride(xShapeInfo), xCoords, xOffset);
    INDEX2COORDS(i, zRank, shape::shapeOf(zShapeInfo), zCoords);
    COORDS2INDEX(zRank, shape::stride(zShapeInfo), zCoords, zOffset);

    z[zOffset] = math::sd_lgamma<T, T>(x[xOffset]);
  }
}

template <typename T>
static void lgamma_(LaunchContext *context, NDArray *x, NDArray *z) {
  auto stream = context->getCudaStream();
  auto length = x->lengthOf();

  dim3 dims = getLaunchDims("lgamma");
  lgammaKernel<T><<<dims.x, dims.y, dims.z, *stream>>>(
      x->specialBuffer(), x->specialShapeInfo(),
      z->specialBuffer(), z->specialShapeInfo(),
      length);

  auto err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess)
    THROW_EXCEPTION("lgamma CUDA kernel failed");
}

void lgamma(LaunchContext *context, NDArray *x, NDArray *z) {
  NDArray::prepareSpecialUse({z}, {x});
  BUILD_SINGLE_SELECTOR(x->dataType(), lgamma_, (context, x, z), SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({z}, {x});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
