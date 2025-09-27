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
// @author AbdelRauf
//
#include <execution/ThreadPool.h>
#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/transforms.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

#if NOT_EXCLUDED(OP_compare_and_bitpack)

namespace sd {
namespace ops {
namespace helpers {

template <typename X>
uint8_t pack(const X* buff, const X& threshold) {
  uint8_t res;
  res = (buff[0] > threshold) << 7;
  res = res | ((buff[1] > threshold) << 6);
  res = res | ((buff[2] > threshold) << 5);
  res = res | ((buff[3] > threshold) << 4);
  res = res | ((buff[4] > threshold) << 3);
  res = res | ((buff[5] > threshold) << 2);
  res = res | ((buff[6] > threshold) << 1);
  res = res | (buff[7] > threshold);
  return res;
}

template <>
uint8_t pack<bool>(const bool* buff, const bool& threshold) {
  // ignore threshold
  uint8_t res;
  res = buff[0] << 7;
  res = res | (buff[1] << 6);
  res = res | (buff[2] << 5);
  res = res | (buff[3] << 4);
  res = res | (buff[4] << 3);
  res = res | (buff[5] << 2);
  res = res | (buff[6] << 1);
  res = res | buff[7];
  return res;
}

template <typename X>
uint8_t pack(const X* buff, int stride, const X& threshold) {
  uint8_t res;
  res = (buff[0] > threshold) << 7;
  res = res | ((buff[1 * stride] > threshold) << 6);
  res = res | ((buff[2 * stride] > threshold) << 5);
  res = res | ((buff[3 * stride] > threshold) << 4);
  res = res | ((buff[4 * stride] > threshold) << 3);
  res = res | ((buff[5 * stride] > threshold) << 2);
  res = res | ((buff[6 * stride] > threshold) << 1);
  res = res | (buff[7 * stride] > threshold);
  return res;
}

template <>
uint8_t pack<bool>(const bool* buff, int stride, const bool& threshold) {
  // ignore threshold
  uint8_t res;
  res = buff[0] << 7;
  res = res | (buff[1 * stride] << 6);
  res = res | (buff[2 * stride] << 5);
  res = res | (buff[3 * stride] << 4);
  res = res | (buff[4 * stride] << 3);
  res = res | (buff[5 * stride] << 2);
  res = res | (buff[6 * stride] << 1);
  res = res | buff[7 * stride];
  return res;
}
template <typename X>
void compareAndBitpack_(NDArray& input, NDArray& thresholdScalar, NDArray& output) {
  auto rank = input.rankOf();
  X threshold = thresholdScalar.e<X>(0);
  auto buff = input.bufferAsT<X>();
  uint8_t* outBuff = output.bufferAsT<uint8_t>();
  if (input.ordering() == 'c' && output.ordering() == 'c' && input.ews() == 1 && output.ews() == 1) {
    FUNC_1D func = [buff, outBuff, threshold](uint64_t thread_id, int64_t start, int64_t stop,
                                              int64_t increment) -> void {
      auto outBuffPart = outBuff + start;
      auto buffPart = buff + start * 8;
      auto len = stop - start;
      // run
      for (auto i = 0; i < len; i++) {
        outBuffPart[i] = pack<X>(&(buffPart[8 * i]), threshold);
      }
    };
    samediff::Threads::parallel_for(func, 0, output.lengthOf(), 1);

  } else {
    auto inShapes = input.shapeOf();
    auto outShapes = output.shapeOf();
    auto inStrides = input.stridesOf();
    auto outStrides = output.stridesOf();

    if (rank == 1) {
      auto inLastStride = inStrides[rank - 1];
      auto outLastStride = outStrides[rank - 1];
      FUNC_1D func = [buff, outBuff, inLastStride, outLastStride, threshold](uint64_t thread_id, int64_t start,
                                                                             int64_t stop, int64_t increment) -> void {
        auto buffPart = buff + start * 8 * inLastStride;
        auto outBuffPart = outBuff + start * outLastStride;
        auto len = stop - start;
        // run
        for (auto i = 0; i < len; i++) {
          *outBuffPart = pack<X>(buffPart, inLastStride, threshold);
          buffPart += 8 * inLastStride;
          outBuffPart += outLastStride;
        }
      };
      samediff::Threads::parallel_for(func, 0, output.lengthOf(), 1);
    } else {
      // if output shape is {n1, n2, n3} then input shape is { n1. n2, n3 * 8}
      // therefore we can split input shape  {n1, n2, n3 , 8} and correct its stride
      // as we do not need last shape info. lets just extend and correct its stride
      sd::LongType extendedStrides[SD_MAX_RANK];
      for (int i = 0; i < rank; i++) {
        extendedStrides[i] = inStrides[i];
      }
      // lets correct new stride
      extendedStrides[rank - 1] = 8 * inStrides[rank - 1];
      extendedStrides[rank] = inStrides[rank - 1];
      // general case. its slow. we can improve it for special case later
      // generic case that could be further improved. for now its slow
      FUNC_1D func = [rank, buff, outBuff, outShapes, extendedStrides, outStrides, threshold](
                         uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
        sd::LongType coords[SD_MAX_RANK] = {};
        sd::LongType* ptr_coords = (sd::LongType*)&coords;
        sd::LongType len = (stop - start);
        // its extended as {rank+1} so extendedStrides[rank] is valid
        auto innermostStride = extendedStrides[rank];
        INDEX2COORDS(start, rank, outShapes, ptr_coords);
        // here last dimension will not be in coords. this way output shape and input shapes are equal
        sd::LongType inOffset, outOffset;
        COORDS2INDEX(rank + 1, extendedStrides, ptr_coords, inOffset);
        COORDS2INDEX(rank, outStrides, ptr_coords, outOffset);
        for (sd::LongType k = 0; k < len; k++) {
          auto buffPart = &(buff[inOffset]);
          auto outBuffPart = &(outBuff[outOffset]);
          *outBuffPart = pack<X>(buffPart, innermostStride, threshold);
          inOffset += extendedStrides[rank];
          outOffset += outStrides[rank - 1];
        }
      };
      samediff::Threads::parallel_for(func, 0, output.lengthOf(), 1);
    }
  }
}

/////////////////////////////////////////////////////////////
void compareAndBitpack(sd::graph::Context& block, NDArray& input, NDArray& threshold, NDArray& output) {
  BUILD_SINGLE_SELECTOR(input.dataType(), compareAndBitpack_, (input, threshold, output), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void compareAndBitpack_,
                      (NDArray& input, NDArray& threshold, NDArray& output), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif