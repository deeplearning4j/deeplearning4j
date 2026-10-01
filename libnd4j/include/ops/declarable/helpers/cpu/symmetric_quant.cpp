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
#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/op_types.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Quantized values land in an aggregate-type staging array, which the central
// NDArray::assign converts once into the quantized data type: the grid bound
// keeps every staged value inside the target's range.
template <typename X, typename S, typename Grid>
static void symmetricQuantize_(Grid, LaunchContext* context, NDArray* input, NDArray* scale, NDArray* quantized) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const AccT bound = symmetricGridBound<AccT>(quantized->dataType());
  std::vector<LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());
  NDArray staging('c', shape, DataTypeUtils::fromT<AccT>(), context);
  NDArray::preparePrimaryUse({&staging}, {input, scale});
  const X* x = input->bufferAsT<X>();
  const S* s = scale->bufferAsT<S>();
  AccT* z = staging.bufferAsT<AccT>();
  const int rank = input->rankOf();
  const int scaleRank = scale->rankOf();
  const LongType* shapeOf = input->shapeOf();
  const LongType* xStrides = input->stridesOf();
  const LongType* zStrides = staging.stridesOf();
  const LongType* sShape = scale->shapeOf();
  const LongType* sStrides = scale->stridesOf();
  auto work = PRAGMA_THREADS_FOR {
    for (LongType linear = start; linear < stop; linear += increment) {
      LongType coords[SD_MAX_RANK];
      INDEX2COORDS(linear, rank, shapeOf, coords);
      LongType xOffset = 0, zOffset = 0;
      COORDS2INDEX(rank, xStrides, coords, xOffset);
      COORDS2INDEX(rank, zStrides, coords, zOffset);
      const LongType sOffset = symmetricScaleOffset(coords, rank, sShape, sStrides, scaleRank);
      z[zOffset] = symmetricQuantizeValue<Grid, AccT>(static_cast<AccT>(x[xOffset]), static_cast<AccT>(s[sOffset]),
                                                      bound);
    }
  };
  samediff::Threads::parallel_for(work, 0, input->lengthOf());
  NDArray::registerPrimaryUse({&staging}, {input, scale});
  quantized->assign(&staging);
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
