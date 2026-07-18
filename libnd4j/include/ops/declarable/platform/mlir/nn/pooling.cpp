/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// MLIR-accelerated pooling operations
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// maxpool2d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(maxpool2d, ENGINE_CPU)

PLATFORM_IMPL(maxpool2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto* output = OUTPUT_VARIABLE(0); // [bS, oH, oW, iC] or [bS, iC, oH, oW]

    // Get pooling parameters
    int kH = INT_ARG(0);  // kernel height
    int kW = INT_ARG(1);  // kernel width
    int sH = INT_ARG(2);  // stride height
    int sW = INT_ARG(3);  // stride width
    int pH = INT_ARG(4);  // padding height
    int pW = INT_ARG(5);  // padding width
    int dH = INT_ARG(6);  // dilation height
    int dW = INT_ARG(7);  // dilation width
    int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME
    int isNCHW = block.numI() > 9 ? INT_ARG(9) : 1;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("maxpool2d", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(maxpool2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR MAXPOOL2D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// avgpool2d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(avgpool2d, ENGINE_CPU)

PLATFORM_IMPL(avgpool2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    int paddingMode = INT_ARG(8);
    int isNCHW = block.numI() > 9 ? INT_ARG(9) : 1;
    int countIncludePad = block.numI() > 10 ? INT_ARG(10) : 0;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("avgpool2d", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(avgpool2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR AVGPOOL2D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// maxpool3d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(maxpool3d, ENGINE_CPU)

PLATFORM_IMPL(maxpool3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iD, iH, iW, iC] or [bS, iC, iD, iH, iW]
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("maxpool3d", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(maxpool3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR MAXPOOL3D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// avgpool3d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(avgpool3d, ENGINE_CPU)

PLATFORM_IMPL(avgpool3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("avgpool3d", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(avgpool3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR AVGPOOL3D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
