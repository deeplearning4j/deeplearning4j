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
// CPU implementations of shape metadata op helpers.
// These write to host buffers directly (standard path for CPU backend).
//

#include <ops/declarable/helpers/shapeOpsHelper.h>
#include <helpers/shape.h>
#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

void shapeOfHelper(LaunchContext* context, NDArray* x, NDArray* z, int rank) {
  const LongType* xShape = shape::shapeOf(x->shapeInfo());

  if (z->getDataBuffer() != nullptr) {
    z->getDataBuffer()->allocatePrimary();
  }

  if (z->dataType() == DataType::INT64) {
    auto zBuff = z->bufferAsT<LongType>();
    std::memcpy(zBuff, xShape, rank * sizeof(LongType));
  } else if (z->dataType() == DataType::INT32) {
    auto zBuff = z->bufferAsT<int>();
    for (int e = 0; e < rank; e++) {
      zBuff[e] = static_cast<int>(xShape[e]);
    }
  } else {
    for (int e = 0; e < rank; e++) {
      z->p(e, xShape[e]);
    }
  }

  z->tickWriteHost();
  z->syncToDevice();
}

void sizeHelper(LaunchContext* context, NDArray* input, NDArray* output) {
  output->p(0, input->lengthOf());
  output->syncToDevice();
}

void rankHelper(LaunchContext* context, NDArray* input, NDArray* output) {
  output->p(0, input->rankOf());
  output->syncToDevice();
}

void sizeAtHelper(LaunchContext* context, NDArray* input, NDArray* output, int dim) {
  output->p(0, input->sizeAt(dim));
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
