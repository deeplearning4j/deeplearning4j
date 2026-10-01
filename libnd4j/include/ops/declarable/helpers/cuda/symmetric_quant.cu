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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_smooth_quant) || NOT_EXCLUDED(OP_fused_norm_quantize)
#include <ops/declarable/helpers/symmetric_quant.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/shape.h>
#include <ops/op_types.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// One element per thread over a grid-stride loop; input, scale and staging are
// addressed through their own strides, so views need no copy.
template <typename X, typename S, typename Grid>
SD_KERNEL static void symmetricQuantizeKernel(const X* input, const LongType* inputShapeInfo, const S* scale,
                                              const LongType* scaleShapeInfo,
                                              typename simdOps::AggregateType<X>::type* staging,
                                              const LongType* stagingShapeInfo,
                                              typename simdOps::AggregateType<X>::type bound) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const int rank = shape::rank(inputShapeInfo);
  const int scaleRank = shape::rank(scaleShapeInfo);
  const LongType* shapeOf = shape::shapeOf(inputShapeInfo);
  const LongType* xStrides = shape::stride(inputShapeInfo);
  const LongType* zStrides = shape::stride(stagingShapeInfo);
  const LongType* sShape = shape::shapeOf(scaleShapeInfo);
  const LongType* sStrides = shape::stride(scaleShapeInfo);
  const LongType length = shape::length(inputShapeInfo);
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; linear < length;
       linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, shapeOf, coords);
    LongType xOffset = 0, zOffset = 0;
    COORDS2INDEX(rank, xStrides, coords, xOffset);
    COORDS2INDEX(rank, zStrides, coords, zOffset);
    const LongType sOffset = symmetricScaleOffset(coords, rank, sShape, sStrides, scaleRank);
    staging[zOffset] = symmetricQuantizeValue<Grid, AccT>(static_cast<AccT>(input[xOffset]),
                                                          static_cast<AccT>(scale[sOffset]), bound);
  }
}

// Quantized values land in an aggregate-type staging array, which the central
// NDArray::assign converts once into the quantized data type: the grid bound
// keeps every staged value inside the target's range. The staging array retires
// behind the assign that consumes it on the stream.
template <typename X, typename S, typename Grid>
static void symmetricQuantize_(Grid, LaunchContext* context, NDArray* input, NDArray* scale, NDArray* quantized) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const AccT bound = symmetricGridBound<AccT>(quantized->dataType());
  std::vector<LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());
  auto* staging = new NDArray('c', shape, DataTypeUtils::fromT<AccT>(), context);
  auto* stream = context->getCudaStream();
  const dim3 dims = getLaunchDims("symmetric_quantize");
  const LongType length = input->lengthOf();
  const LongType needed = (length - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  NDArray::prepareSpecialUse({staging}, {input, scale});
  symmetricQuantizeKernel<X, S, Grid><<<blocks, dims.y, dims.z, *stream>>>(
      static_cast<const X*>(input->specialBuffer()), input->specialShapeInfo(),
      static_cast<const S*>(scale->specialBuffer()), scale->specialShapeInfo(),
      static_cast<AccT*>(staging->specialBuffer()), staging->specialShapeInfo(), bound);
  NDArray::registerSpecialUse({staging}, {input, scale});
  DebugHelper::checkGlobalErrorCode("symmetricQuantize launch failed");
  quantized->assign(staging);
  MmulHelper::deleteTemporary(staging);
}

void symmetricQuantize(LaunchContext* context, NDArray* input, NDArray* scale, NDArray* quantized) {
  symmetricQuantizeCheck(input, scale, quantized);
  if (input->isEmpty()) return;
  if (DataTypeUtils::isZ(quantized->dataType())) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), scale->dataType(), symmetricQuantize_,
                          (SymmetricIntegerGrid{}, context, input, scale, quantized), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  } else {
    BUILD_DOUBLE_SELECTOR(input->dataType(), scale->dataType(), symmetricQuantize_,
                          (SymmetricFloatingGrid{}, context, input, scale, quantized), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
