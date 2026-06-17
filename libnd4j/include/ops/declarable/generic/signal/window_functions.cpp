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
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_hann_window)

#include <ops/declarable/headers/signal.h>
#include <ops/declarable/helpers/signal.h>

namespace sd {
namespace ops {

// ===================== HANN WINDOW =====================

CUSTOM_OP_IMPL(hann_window, 1, 1, false, 0, -2) {
    auto sizeInput = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int size = sizeInput->e<int>(0);
    bool periodic = block.numI() > 0 ? INT_ARG(0) != 0 : true;

    helpers::hannWindow(block.launchContext(), size, periodic, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(hann_window) {
    auto sizeInput = INPUT_VARIABLE(0);
    sd::LongType size = sizeInput->e<sd::LongType>(0);

    std::vector<sd::LongType> outputShapeVec = {size};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::FLOAT32, 'c', outputShapeVec));
}

DECLARE_TYPES(hann_window) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif

// ===================== HAMMING WINDOW =====================

#if NOT_EXCLUDED(OP_hamming_window)

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(hamming_window, 1, 1, false, 0, -2) {
    auto sizeInput = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int size = sizeInput->e<int>(0);
    bool periodic = block.numI() > 0 ? INT_ARG(0) != 0 : true;

    helpers::hammingWindow(block.launchContext(), size, periodic, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(hamming_window) {
    auto sizeInput = INPUT_VARIABLE(0);
    sd::LongType size = sizeInput->e<sd::LongType>(0);

    std::vector<sd::LongType> outputShapeVec = {size};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::FLOAT32, 'c', outputShapeVec));
}

DECLARE_TYPES(hamming_window) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif

// ===================== BLACKMAN WINDOW =====================

#if NOT_EXCLUDED(OP_blackman_window)

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(blackman_window, 1, 1, false, 0, -2) {
    auto sizeInput = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int size = sizeInput->e<int>(0);
    bool periodic = block.numI() > 0 ? INT_ARG(0) != 0 : true;

    helpers::blackmanWindow(block.launchContext(), size, periodic, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(blackman_window) {
    auto sizeInput = INPUT_VARIABLE(0);
    sd::LongType size = sizeInput->e<sd::LongType>(0);

    std::vector<sd::LongType> outputShapeVec = {size};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::FLOAT32, 'c', outputShapeVec));
}

DECLARE_TYPES(blackman_window) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
