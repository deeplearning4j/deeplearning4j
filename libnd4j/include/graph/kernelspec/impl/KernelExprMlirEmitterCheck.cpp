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

// Compile-verification anchor for the header-only MLIR interpreter.
// HAVE_MLIR arrives as a command-line definition on MLIR-enabled builds
// (Dependencies.cmake add_compile_definitions), same gate CpuIRBuilder.cpp
// uses; on non-MLIR builds this TU is intentionally empty.
#if HAVE_MLIR

#include <graph/kernelspec/KernelExprMlirEmitter.h>

namespace sd {
namespace kernelspec {

// Taking the functions' addresses forces full codegen of the inline bodies so
// MLIR API drift is caught at build time even while nothing is wired.
const void* kernelExprMlirEmitterAnchor() {
  static const auto emitPtr = &emitKernelExpr;
  static const auto policyPtr = &makeDefaultMlirEmitPolicy;
  static const auto splatPtr = &kspecSplatConstant;
  return emitPtr && policyPtr && splatPtr ? reinterpret_cast<const void*>(emitPtr) : nullptr;
}

}  // namespace kernelspec
}  // namespace sd

#endif  // HAVE_MLIR
