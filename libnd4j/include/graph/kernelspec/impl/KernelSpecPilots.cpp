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

#include <graph/kernelspec/KernelSpec.h>
#include <ops/declarable/OpDescriptor.h>

#include <limits>
#include <mutex>
#include <stdexcept>

// Shared production expression specs (ADR-0116). Runtime emitters resolve them
// by canonical OpDescriptor hash; names and aliases are authoring/diagnostic
// metadata only. Registration is explicit and idempotent.

namespace sd {
namespace kernelspec {

namespace {

void registerAll() {
  using ops::OP_TRAIT_ACTIVATION;
  using ops::OP_TRAIT_BINARY_ELEMENTWISE;
  using ops::OP_TRAIT_FULLY_WRITING;
  using ops::OP_TRAIT_REDUCTION;
  using ops::OP_TRAIT_UNARY_ELEMENTWISE;

  std::string err;
  auto require = [&err](bool ok) {
    if (ok) return;
    // Selective-op builds legitimately omit descriptors. An absent canonical op
    // means this spec is unavailable in that artifact, not that registration is
    // malformed. Any other failure remains fatal.
    if (err.find("no canonical registered operation") != std::string::npos) {
      err.clear();
      return;
    }
    throw std::logic_error("kernelspec builtin registration failed: " + err);
  };

  constexpr uint32_t kFloatTypes = KDT_F32 | KDT_F16 | KDT_BF16 | KDT_F64;

  require(KernelSpecBuilder("add")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return g.input(0) + g.input(1); })
              .registerSpec(&err));
  require(KernelSpecBuilder("subtract")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return g.input(0) - g.input(1); })
              .registerSpec(&err));
  require(KernelSpecBuilder("multiply")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return g.input(0) * g.input(1); })
              .registerSpec(&err));
  require(KernelSpecBuilder("divide")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return g.input(0) / g.input(1); })
              .registerSpec(&err));
  require(KernelSpecBuilder("maximum")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return max(g.input(0), g.input(1)); })
              .registerSpec(&err));
  require(KernelSpecBuilder("minimum")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return min(g.input(0), g.input(1)); })
              .registerSpec(&err));
  require(KernelSpecBuilder("Pow")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return pow(g.input(0), g.input(1)); })
              .registerSpec(&err));
  require(KernelSpecBuilder("tanh")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING |
                      OP_TRAIT_ACTIVATION)
              .dtypes(kFloatTypes)
              .body([](ExprGraph& g) { return tanh(g.input(0)); })
              .registerSpec(&err));
  require(KernelSpecBuilder("sigmoid")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING |
                      OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .body([](ExprGraph& g) { return sigmoid(g.input(0)); })
              .registerSpec(&err));
  require(KernelSpecBuilder("relu")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING |
                      OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .scalar("threshold", 0, 0.0)
              .body([](ExprGraph& g) {
                return max(g.input(0), g.scalarParam(0));
              })
              .registerSpec(&err));

  // SwiGLU gate: must match TritonIRBuilder_emitters.cpp "custom.swish_mul".
  require(KernelSpecBuilder("swish_mul")
              .alias("SwishMul")
              .category(KernelCategory::BINARY_ELEMENTWISE)
              .traits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .body([](ExprGraph& g) {
                auto x = g.input(0);
                auto y = g.input(1);
                return silu(x) * y;
              })
              .registerSpec(&err));

  require(KernelSpecBuilder("elu")
              .alias("Elu")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .scalar("alpha", 0, 1.0)
              .body([](ExprGraph& g) {
                auto x = g.input(0);
                auto alpha = g.scalarParam(0);
                return select(x > 0.0, x, alpha * (exp(x) - 1.0));
              })
              .registerSpec(&err));

  require(KernelSpecBuilder("clipbyvalue")
              .alias("ClipByValue")
              .alias("clip_by_value")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16 | KDT_F64)
              .scalar("clipValueMin", 0, -1.0)
              .scalar("clipValueMax", 1, 1.0)
              .body([](ExprGraph& g) {
                auto x = g.input(0);
                return clamp(x, g.scalarParam(0), g.scalarParam(1));
              })
              .registerSpec(&err));

  require(KernelSpecBuilder("hardsigmoid")
              .alias("HardSigmoid")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .body([](ExprGraph& g) { return hardSigmoid(g.input(0)); })
              .registerSpec(&err));

  require(KernelSpecBuilder("mish")
              .alias("Mish")
              .category(KernelCategory::UNARY_ELEMENTWISE)
              .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16)
              .body([](ExprGraph& g) { return mish(g.input(0)); })
              .registerSpec(&err));

  // Reductions as (init, combine, finalize) triples — the same decomposition
  // Vulkan's reductionCallbacksFor and Triton's combiner regions already use.
  require(KernelSpecBuilder("reduce_sum")
              .alias("ReduceSum")
              .category(KernelCategory::REDUCTION)
              .traits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16 | KDT_F64)
              .reduction([](ExprGraph& g) { return g.c(0.0); },
                         [](ExprGraph& g) { return g.input(0) + g.input(1); },
                         [](ExprGraph& g) { return g.input(0); })
              .registerSpec(&err));

  require(KernelSpecBuilder("reduce_mean")
              .alias("ReduceMean")
              .category(KernelCategory::REDUCTION)
              .traits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16 | KDT_F64)
              .reduction([](ExprGraph& g) { return g.c(0.0); },
                         [](ExprGraph& g) { return g.input(0) + g.input(1); },
                         [](ExprGraph& g) { return g.input(0) / g.input(1); })
              .registerSpec(&err));

  require(KernelSpecBuilder("reduce_max")
              .alias("ReduceMax")
              .category(KernelCategory::REDUCTION)
              .traits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING)
              .dtypes(KDT_F32 | KDT_F16 | KDT_BF16 | KDT_F64)
              .reduction([](ExprGraph& g) { return g.c(-std::numeric_limits<double>::infinity()); },
                         [](ExprGraph& g) { return max(g.input(0), g.input(1)); },
                         [](ExprGraph& g) { return g.input(0); })
              .registerSpec(&err));
}

}  // namespace

void registerBuiltinKernelSpecs() {
  static std::once_flag once;
  std::call_once(once, registerAll);
}

void registerPilotKernelSpecs() { registerBuiltinKernelSpecs(); }

}  // namespace kernelspec
}  // namespace sd
