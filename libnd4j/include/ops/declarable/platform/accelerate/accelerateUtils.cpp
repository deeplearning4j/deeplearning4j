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
// Apple Accelerate framework helper utilities implementation
//

#include "accelerateUtils.h"
#include <ConstMessages.h>
#include <system/Environment.h>

namespace sd {
namespace ops {
namespace accelerateUtils {

bool isAccelerateSupported(sd::DataType dtype) {
    // Accelerate framework primarily supports float32 and float64
    // BNNS also supports float16 and some integer types
    switch (dtype) {
        case sd::DataType::FLOAT32:
        case sd::DataType::DOUBLE:
        case sd::DataType::FLOAT16:
            return true;
        default:
            return false;
    }
}

bool isContiguous(const NDArray& arr) {
    auto rank = arr.rankOf();
    auto shape = arr.shapeOf();
    auto strides = arr.stridesOf();
    sd::LongType expected = 1;
    for (int i = rank - 1; i >= 0; --i) {
      if (shape[i] == 1) continue;
      if (strides[i] != expected) return false;
      expected *= shape[i];
    }
    return true;
}

bool isAccelerateFriendly(const NDArray& arr) {
    // Array must be contiguous and have a supported data type
    if (!isContiguous(arr)) {
        return false;
    }

    if (!isAccelerateSupported(arr.dataType())) {
        return false;
    }

    // Array must not be empty
    if (arr.isEmpty()) {
        return false;
    }

    return true;
}

#ifdef HAVE_ACCELERATE
BNNSDataType getBNNSDataType(sd::DataType dtype) {
    switch (dtype) {
        case sd::DataType::FLOAT32:
            return BNNSDataTypeFloat32;
        case sd::DataType::FLOAT16:
            return BNNSDataTypeFloat16;
        case sd::DataType::INT8:
            return BNNSDataTypeInt8;
        case sd::DataType::INT16:
            return BNNSDataTypeInt16;
        case sd::DataType::INT32:
            return BNNSDataTypeInt32;
        case sd::DataType::UINT8:
            return BNNSDataTypeUInt8;
        case sd::DataType::UINT16:
            return BNNSDataTypeUInt16;
        case sd::DataType::UINT32:
            return BNNSDataTypeUInt32;
        default:
            // Default to float32 for unsupported types
            return BNNSDataTypeFloat32;
    }
}
#endif

void checkAccelerateRequirements(Requirements& reqs, sd::graph::Context& block,
                                  const NDArray* input, const NDArray* output) {
    // Check if Accelerate is enabled
    reqs.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);

    // Check input array if provided
    if (input != nullptr) {
        reqs.expectTrue(isAccelerateFriendly(*input),
                        "Input array must be contiguous with supported data type");
    }

    // Check output array if provided
    if (output != nullptr) {
        reqs.expectTrue(isAccelerateFriendly(*output),
                        "Output array must be contiguous with supported data type");
    }
}

bool canUseAccelerateBlas(const NDArray& a, const NDArray* b, const NDArray* c) {
    // Check if array 'a' is suitable
    if (!isAccelerateFriendly(a)) {
        return false;
    }

    // For BLAS operations, we primarily support float32 and float64
    if (a.dataType() != sd::DataType::FLOAT32 && a.dataType() != sd::DataType::DOUBLE) {
        return false;
    }

    // Check array 'b' if provided
    if (b != nullptr) {
        if (!isAccelerateFriendly(*b)) {
            return false;
        }
        // Data types must match
        if (a.dataType() != b->dataType()) {
            return false;
        }
    }

    // Check array 'c' if provided
    if (c != nullptr) {
        if (!isAccelerateFriendly(*c)) {
            return false;
        }
        // Data types must match
        if (a.dataType() != c->dataType()) {
            return false;
        }
    }

    return true;
}

}  // namespace accelerateUtils
}  // namespace ops
}  // namespace sd
