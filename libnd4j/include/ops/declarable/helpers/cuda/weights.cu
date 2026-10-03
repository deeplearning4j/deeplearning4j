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
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bincount)
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/weights.h>
#include <ops/op_types.h>

#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {

// sums[v] += weights[i] (or 1) for each value v = values[i] in [0, bins); other values add nothing. values and
// weights (same shape) are read through their own strides; sums is a dense accumulator.
template <typename T, typename AccT>
static SD_KERNEL void bincountKernel(const LongType* values, const LongType* valuesShapeInfo, const T* weights,
                                     const LongType* weightsShapeInfo, AccT* sums, const LongType bins) {
  const LongType length = shape::length(valuesShapeInfo);
  const int rank = shape::rank(valuesShapeInfo);
  const LongType* shape = shape::shapeOf(valuesShapeInfo);
  const LongType* valueStrides = shape::stride(valuesShapeInfo);
  const LongType* weightStrides = weights != nullptr ? shape::stride(weightsShapeInfo) : nullptr;

  LongType coords[SD_MAX_RANK];
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    INDEX2COORDS(i, rank, shape, coords);
    LongType valueOffset;
    COORDS2INDEX(rank, valueStrides, coords, valueOffset);
    const LongType v = values[valueOffset];
    if (v < 0 || v >= bins) continue;
    AccT add = static_cast<AccT>(1);
    if (weights != nullptr) {
      LongType weightOffset;
      COORDS2INDEX(rank, weightStrides, coords, weightOffset);
      add = static_cast<AccT>(weights[weightOffset]);
    }
    math::atomics::sd_atomicAdd<AccT>(sums + v, add);
  }
}

// input holds INT64 values and weights, when given, the output's type and the input's shape
template <typename T>
static void adjustWeights_(LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output) {
  // Sums in a wider type than the bins hold: 64-bit integers, or FLOAT for HALF and BFLOAT16
  using AccT = typename std::conditional<std::is_integral<T>::value, LongType,
                                         typename simdOps::AggregateType<T>::type>::type;
  const LongType bins = output->lengthOf();
  if (bins == 0) return;
  if (input->lengthOf() == 0) {
    output->nullify();
    return;
  }

  std::vector<LongType> sumsShape = {bins};
  NDArray sums('c', sumsShape, DataTypeUtils::fromT<AccT>(), context);
  sums.nullify();

  dim3 launchDims = getLaunchDims("adjustWeights");
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({&sums}, {input, weights});
  bincountKernel<T, AccT><<<launchDims.x, launchDims.y, 0, *stream>>>(
      reinterpret_cast<const LongType*>(input->specialBuffer()), input->specialShapeInfo(),
      weights != nullptr ? reinterpret_cast<const T*>(weights->specialBuffer()) : nullptr,
      weights != nullptr ? weights->specialShapeInfo() : nullptr, reinterpret_cast<AccT*>(sums.specialBuffer()), bins);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("bincountKernel failed");
  }
  NDArray::registerSpecialUse({&sums}, {input, weights});

  output->assign(&sums);

  // sums is released when this returns: wait for the kernel and the assign that read it
  PointersManager manager(context, "bincount");
  manager.synchronize();
}

void adjustWeights(LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                   int maxLength) {
  BUILD_SINGLE_SELECTOR(output->dataType(), adjustWeights_, (context, input, weights, output), SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE(void adjustWeights_, (sd::LaunchContext * context, NDArray* input, NDArray* weights,
                                            NDArray* output),
                      SD_NUMERIC_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
