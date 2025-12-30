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

#ifndef SD_OPS_DECLARABLE_PLATFORM_MLIR_MLIRUTILS_H
#define SD_OPS_DECLARABLE_PLATFORM_MLIR_MLIRUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <system/RequirementsHelper.h>

#ifdef HAVE_MLIR
#include <mlir/runtime/MLIREngine.h>
#endif

#include <vector>
#include <string>

namespace sd {
namespace ops {
namespace platforms {

/// Check if MLIR JIT is available and enabled for this build
inline bool mlirEnabled() {
#ifdef HAVE_MLIR
    return true;
#else
    return false;
#endif
}

/// Check if MLIR JIT engine is initialized
inline bool mlirInitialized() {
#ifdef HAVE_MLIR
    return mlir_runtime::MLIREngine::getInstance().isInitialized();
#else
    return false;
#endif
}

/// Initialize MLIR JIT engine if not already done
inline bool initializeMlir() {
#ifdef HAVE_MLIR
    return mlir_runtime::MLIREngine::getInstance().initialize();
#else
    return false;
#endif
}

/// Get shapes from NDArray for MLIR compilation
inline std::vector<int64_t> getShape(const NDArray* arr) {
    std::vector<int64_t> shape;
    if (arr != nullptr) {
        auto shapeInfo = arr->shapeInfo();
        int rank = arr->rankOf();
        shape.reserve(rank);
        for (int i = 0; i < rank; ++i) {
            shape.push_back(arr->sizeAt(i));
        }
    }
    return shape;
}

/// Get shapes from context inputs for MLIR compilation
inline std::vector<std::vector<int64_t>> getInputShapes(graph::Context& context) {
    std::vector<std::vector<int64_t>> shapes;
    int numInputs = context.width();
    shapes.reserve(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        auto* input = context.getNDArray(i);
        shapes.push_back(getShape(input));
    }
    return shapes;
}

/// Get data types from context inputs for MLIR compilation
inline std::vector<int> getInputTypes(graph::Context& context) {
    std::vector<int> types;
    int numInputs = context.width();
    types.reserve(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        auto* input = context.getNDArray(i);
        if (input != nullptr) {
            types.push_back(static_cast<int>(input->dataType()));
        } else {
            types.push_back(0);
        }
    }
    return types;
}

/// Minimum tensor size threshold for MLIR JIT to be beneficial
/// Below this threshold, the JIT overhead exceeds the benefit
constexpr int64_t MLIR_MIN_TENSOR_SIZE = 1024;

/// Check if tensor size meets minimum threshold for MLIR
inline bool meetsSizeThreshold(const NDArray* arr, int64_t minSize = MLIR_MIN_TENSOR_SIZE) {
    return arr != nullptr && arr->lengthOf() >= minSize;
}

/// Check if any input meets size threshold
inline bool anyInputMeetsSizeThreshold(graph::Context& context, int64_t minSize = MLIR_MIN_TENSOR_SIZE) {
    int numInputs = context.width();
    for (int i = 0; i < numInputs; ++i) {
        auto* input = context.getNDArray(i);
        if (meetsSizeThreshold(input, minSize)) {
            return true;
        }
    }
    return false;
}

/// Supported data types for MLIR operations
inline bool isSupportedMlirType(DataType dtype) {
    switch (dtype) {
        case FLOAT32:
        case DOUBLE:
        case HALF:
        case BFLOAT16:
        case INT32:
        case INT64:
            return true;
        default:
            return false;
    }
}

/// Check if all inputs have supported MLIR data types
inline bool allInputsSupportedMlirType(graph::Context& context) {
    int numInputs = context.width();
    for (int i = 0; i < numInputs; ++i) {
        auto* input = context.getNDArray(i);
        if (input != nullptr && !isSupportedMlirType(input->dataType())) {
            return false;
        }
    }
    return true;
}

#ifdef HAVE_MLIR

/// Execute operation via MLIR JIT
/// @param opName Name of the operation
/// @param context Operation context
/// @param inputs Input NDArrays
/// @param outputs Output NDArrays
/// @return Status::OK on success
inline Status executeMlir(
    const std::string& opName,
    graph::Context& context,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {

    auto& engine = mlir_runtime::MLIREngine::getInstance();

    if (!engine.isInitialized() && !engine.initialize()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "Failed to initialize MLIR engine");
    }

    // Get input shapes and types
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<int> inputTypes;
    for (auto* input : inputs) {
        inputShapes.push_back(getShape(input));
        inputTypes.push_back(static_cast<int>(input->dataType()));
    }

    // Get or compile kernel
    auto kernel = engine.getOrCompile(opName, inputShapes, inputTypes);
    if (!kernel || !kernel->isValid()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "Failed to compile MLIR kernel for " + opName);
    }

    // Execute kernel
    if (!kernel->execute(inputs, outputs)) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "Failed to execute MLIR kernel for " + opName);
    }

    return Status::OK();
}

#endif // HAVE_MLIR

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // SD_OPS_DECLARABLE_PLATFORM_MLIR_MLIRUTILS_H
