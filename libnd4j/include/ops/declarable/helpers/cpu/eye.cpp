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
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_eye)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void eye_(NDArray& output) {
  const int rank = output.rankOf();
  const int batchRank = rank - 2;
  const auto diagLen = sd::math::sd_min(output.sizeAt(rank - 2), output.sizeAt(rank - 1));
  const auto* strides = shape::stride(output.shapeInfo());

  output.nullify();
  if (diagLen <= 0) {
    return;
  }

  const T one = static_cast<T>(1);
  auto* buffer = output.bufferAsT<T>();
  if (batchRank <= 0) {
    for (sd::LongType i = 0; i < diagLen; ++i) {
      const auto offset = i * strides[rank - 2] + i * strides[rank - 1];
      buffer[offset] = one;
    }
    output.tickWriteHost();
    output.syncToDevice();
    return;
  }

  std::vector<sd::LongType> batchShape(batchRank);
  sd::LongType batchCount = 1;
  for (int d = 0; d < batchRank; ++d) {
    batchShape[d] = output.sizeAt(d);
    batchCount *= batchShape[d];
  }

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType batchCoords[SD_MAX_RANK];
    for (auto batch = start; batch < stop; ++batch) {
      INDEX2COORDS(batch, batchRank, batchShape.data(), batchCoords);

      sd::LongType batchOffset = 0;
      for (int d = 0; d < batchRank; ++d) {
        batchOffset += batchCoords[d] * strides[d];
      }

      for (sd::LongType i = 0; i < diagLen; ++i) {
        const auto offset = batchOffset + i * strides[rank - 2] + i * strides[rank - 1];
        buffer[offset] = one;
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, batchCount);
  output.tickWriteHost();
  output.syncToDevice();
}

void eye(sd::LaunchContext* context, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), eye_, (output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
