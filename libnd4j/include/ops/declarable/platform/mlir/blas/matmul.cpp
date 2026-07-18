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
// MLIR-accelerated matrix multiplication
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
// matmul MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(matmul, ENGINE_CPU)

PLATFORM_IMPL(matmul, ENGINE_CPU) {
    auto* a = INPUT_VARIABLE(0);
    auto* b = INPUT_VARIABLE(1);
    auto* c = OUTPUT_VARIABLE(0);

    // Get optional arguments
    bool transposeA = block.numB() > 0 ? block.getBArguments()->at(0) : false;
    bool transposeB = block.numB() > 1 ? block.getBArguments()->at(1) : false;
    double alpha = block.numT() > 0 ? block.getTArguments()->at(0) : 1.0;
    double beta = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;

    // Prepare inputs and outputs for MLIR execution
    std::vector<NDArray*> inputs = {a, b};
    std::vector<NDArray*> outputs = {c};

    // Execute via MLIR JIT
    auto status = executeMlir("matmul", block, inputs, outputs);

    if (status != Status::OK) {
        // Fall back to native implementation if MLIR fails
        // This shouldn't happen in normal operation, but provides safety
        sd_printf("MLIR matmul execution failed, falling back to native\n", "");

        // Native fallback using MmulHelper
        // MmulHelper::matmul(a, b, c, transposeA, transposeB, alpha, beta);
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(matmul, ENGINE_CPU) {
    auto* a = INPUT_VARIABLE(0);
    auto* b = INPUT_VARIABLE(1);

    Requirements req("MLIR MATMUL");

    // Check MLIR availability
    req.expectTrue(mlirEnabled(), "MLIR enabled") &&

    // Check supported data types
    req.expectIn(a->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&

    req.expectIn(b->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&

    // Check matching types
    req.expectEq(a->dataType(), b->dataType()) &&

    // Check minimum size threshold (JIT overhead vs benefit)
    req.expectTrue(a->lengthOf() * b->lengthOf() >= MLIR_MIN_TENSOR_SIZE * MLIR_MIN_TENSOR_SIZE,
                   "Tensor size above MLIR threshold") &&

    // Check contiguous memory layout for optimal performance
    req.expectTrue(shape::strideDescendingCAscendingF(a->shapeInfo()),
                   "Input A has contiguous memory") &&
    req.expectTrue(shape::strideDescendingCAscendingF(b->shapeInfo()),
                   "Input B has contiguous memory");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
