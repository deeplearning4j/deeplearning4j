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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_compat_sparse_to_dense)
#include <helpers/ShapeUtils.h>
#include <helpers/StringUtils.h>
#include <ops/declarable/helpers/sparse_to_dense.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename X, typename I>
static void fill_(const void *vvalues, const void *vindices, void *voutput, const LongType *zShapeInfo,
                  uint8_t rank, uint64_t length) {
  auto values = reinterpret_cast<const X *>(vvalues);
  auto indices = reinterpret_cast<const I *>(vindices);
  auto output = reinterpret_cast<X *>(voutput);

  LongType coords[SD_MAX_RANK];
  uint64_t pos = 0;
  for (uint64_t e = 0L; e < length; e++) {
    // indices come in blocks
    for (uint8_t p = 0; p < rank; p++) {
      coords[p] = indices[pos++];
    }

    // fill output at given coords with sparse value
    LongType offset;
    COORDS2INDEX(rank, shape::stride(zShapeInfo), coords, offset);
    output[offset] = values[e];
  }
}
void compat_sparse_to_dense(NDArray& values, NDArray& indices, NDArray* def, NDArray& output) {
  // make sure host buffer is updated

  auto rank = output.rankOf();

  if (output.isS()) {
    NDArray::preparePrimaryUse({&output}, {&values, &indices, def});
    // string case is not so trivial, since elements might, and probably will, have different sizes
    auto numValues = values.lengthOf();
    auto numElements = output.lengthOf();

    // first of all we calculate final buffer sizes and offsets
    auto defaultLength = def == nullptr ? 0 : StringUtils::byteLength(*def);
    auto valuesLength = StringUtils::byteLength(values);
    auto bufferLength = defaultLength * (output.lengthOf() - numValues) + valuesLength;
    auto headerLength = ShapeUtils::stringBufferHeaderRequirements(numElements);

    // now we make sure our output buffer can hold results
    output.dataBuffer()->expand(bufferLength + headerLength);

    std::vector<LongType> outputCoords(rank);
    std::vector<LongType> valueCoords(rank);

    auto offsetsBuffer = output.bufferAsT<LongType>();
    auto dataBuffer = reinterpret_cast<uint8_t*>(offsetsBuffer + output.lengthOf());

    offsetsBuffer[0] = 0;

    // getting initial value coords
    for (int e = 0; e < rank; e++) valueCoords[e] = indices.e<LongType>(e);

    // write results individually
    for (LongType e = 0; e < numElements; e++) {
      LongType vIndex;
      COORDS2INDEX(rank, shape::stride(output.shapeInfo()), valueCoords.data(), vIndex);
      auto cLength = 0L;
      std::string str;
      if (vIndex == e) {
        // we're writing down sparse value here
        str = values.e<std::string>(e);
      } else {
        // we're writing down default value if it exists
        if (def != nullptr)
          str = def->e<std::string>(0);
        else
          str = "";
      }

      // TODO: make it unicode compliant
      memcpy(&dataBuffer[offsetsBuffer[e]], str.c_str(), str.length());

      // writing down offset
      offsetsBuffer[e + 1] = cLength;
    }
    NDArray::registerPrimaryUse({&output}, {&values, &indices, def});
  } else {
    // numeric case is trivial, since all elements have equal sizes

    // write out default values, if they are present
    if (def != nullptr) {
      output.assign(def);
    }
    NDArray::preparePrimaryUse({&output}, {&values, &indices});
    // write out values
    BUILD_DOUBLE_SELECTOR(
        values.dataType(), indices.dataType(), fill_,
        (values.buffer(), indices.buffer(), output.buffer(), output.shapeInfo(), rank, values.lengthOf()),
        SD_COMMON_TYPES, SD_INDEXING_TYPES);

    NDArray::registerPrimaryUse({&output}, {&values, &indices});
  }
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif