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

// Pilot kernel specs (ADR-0116). These mirror ops that already exist in the
// hand-written emitters so the consistency checker can cross-validate the DSL
// against the live catalogs. Registration is explicit and idempotent — no
// static initializer runs it, and no execution path consumes it yet.

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
    if (!ok) throw std::logic_error("kernelspec pilot registration failed: " + err);
  };

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

void registerPilotKernelSpecs() {
  static std::once_flag once;
  std::call_once(once, registerAll);
}

}  // namespace kernelspec
}  // namespace sd
