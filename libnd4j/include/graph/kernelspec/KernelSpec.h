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

#ifndef LIBND4J_KERNELSPEC_KERNELSPEC_H
#define LIBND4J_KERNELSPEC_KERNELSPEC_H

#include <graph/kernelspec/KernelExpr.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Single-source kernel specification (ADR-0116). One KernelSpec carries the
// metadata that today is hand-synced across five parallel catalogs
// (addTraits/OpTraitTable, OpCategoryTable, Triton buildOpTable,
// VulkanKernelEmitterCatalog, the Java trait mirror) plus the op's math as a
// KernelExpr body or reduction triple.
//
// NOT wired: no emitter or table consults this registry yet. The registry is
// the future single source those tables are generated from / checked against.

namespace sd {
namespace kernelspec {

// Authorable computation shapes, v1. Maps 1:1 onto the shared
// TritonOpCategory values of the same name (see KernelSpecConsistency.h);
// structural categories (matmul, attention, normalization, data movement)
// stay hand-written recipes and are deliberately absent.
enum class KernelCategory : uint8_t {
  UNARY_ELEMENTWISE = 0,
  BINARY_ELEMENTWISE,
  TERNARY_ELEMENTWISE,
  COMPARISON,
  LOGICAL,
  REDUCTION,
  IDENTITY
};

const char* kernelCategoryName(KernelCategory c);

// Backend-neutral dtype capability mask; mapped to sd::DataType and to
// per-backend capability gates (e.g. Vulkan fp16/storage16 caps) at wiring.
enum KernelDTypeMask : uint32_t {
  KDT_F32 = 1u << 0,
  KDT_F16 = 1u << 1,
  KDT_BF16 = 1u << 2,
  KDT_F64 = 1u << 3,
  KDT_I32 = 1u << 4,
  KDT_I64 = 1u << 5,
  KDT_I8 = 1u << 6,
  KDT_BOOL = 1u << 7
};

// A tArgs-backed scalar parameter (e.g. elu alpha = tArgs[0]).
struct ScalarParamSpec {
  std::string name;
  int tArgIndex = 0;
  double defaultValue = 0.0;
};

// Reduction as an (init, combine, finalize) triple.
// Conventions the validator enforces and interpreters rely on:
//   init:     no INPUT leaves (constants / scalar params only)
//   combine:  INPUT(0) = accumulator, INPUT(1) = element
//   finalize: INPUT(0) = accumulator, INPUT(1) = element count (optional use)
struct ReductionTripleSpec {
  ExprGraph init;
  ExprGraph combine;
  ExprGraph finalize;
  bool present = false;
};

struct KernelSpec {
  std::string name;
  std::vector<std::string> aliases;  // e.g. the CamelCase table variants
  KernelCategory category = KernelCategory::UNARY_ELEMENTWISE;
  uint32_t traits = 0;  // OP_TRAIT_* bitmask; constants live in ops/declarable/OpDescriptor.h
  uint32_t dtypes = KDT_F32;
  int numInputs = 1;
  std::vector<ScalarParamSpec> scalars;
  ExprGraph body;  // elementwise/comparison/ternary/identity body
  bool hasBody = false;
  ReductionTripleSpec reduction;
  // Engines where an existing hand-written kernel must keep shadowing the
  // DSL emission (standard override-beats-emitter precedence).
  std::vector<std::string> handWrittenOverride;
  std::string notes;

  // Empty string when valid, otherwise the first problem found.
  std::string validate() const;
};

class KernelSpecRegistry {
 public:
  static KernelSpecRegistry& getInstance();

  // Validates and inserts. Returns false (and fills *error when given) on
  // validation failure or duplicate name/alias.
  bool add(const KernelSpec& spec, std::string* error = nullptr);

  // Lookup by primary name or alias; nullptr when absent.
  const KernelSpec* find(const std::string& nameOrAlias) const;

  std::vector<const KernelSpec*> all() const;
  size_t size() const;

 private:
  KernelSpecRegistry() = default;
  struct Impl;
  Impl& impl() const;
};

// Fluent authoring surface.
//
//   KernelSpecBuilder("elu")
//       .alias("Elu")
//       .category(KernelCategory::UNARY_ELEMENTWISE)
//       .traits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING)
//       .dtypes(KDT_F32 | KDT_F16)
//       .scalar("alpha", 0, 1.0)
//       .body([](ExprGraph& g) {
//         auto x = g.input(0);
//         auto alpha = g.scalarParam(0);
//         return select(x > 0.0, x, alpha * (exp(x) - 1.0));
//       })
//       .registerSpec();
class KernelSpecBuilder {
 public:
  explicit KernelSpecBuilder(std::string name);

  KernelSpecBuilder& alias(std::string a);
  KernelSpecBuilder& category(KernelCategory c);
  KernelSpecBuilder& traits(uint32_t traitMask);
  KernelSpecBuilder& dtypes(uint32_t dtypeMask);
  KernelSpecBuilder& inputs(int n);
  KernelSpecBuilder& scalar(std::string name, int tArgIndex, double defaultValue);
  KernelSpecBuilder& body(const std::function<Expr(ExprGraph&)>& build);
  KernelSpecBuilder& reduction(const std::function<Expr(ExprGraph&)>& init,
                               const std::function<Expr(ExprGraph&)>& combine,
                               const std::function<Expr(ExprGraph&)>& finalize);
  KernelSpecBuilder& handWritten(std::string engine);
  KernelSpecBuilder& notes(std::string text);

  // Validate + insert into the global registry.
  bool registerSpec(std::string* error = nullptr);
  // Validate + return the spec without registering (for tests / generators).
  KernelSpec build() const;

 private:
  KernelSpec spec_;
};

}  // namespace kernelspec
}  // namespace sd

#endif  // LIBND4J_KERNELSPEC_KERNELSPEC_H
