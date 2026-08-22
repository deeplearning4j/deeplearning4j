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
#include <ops/declarable/OpRegistrator.h>

#include <map>
#include <mutex>
#include <set>

namespace sd {
namespace kernelspec {

const char* kernelCategoryName(KernelCategory c) {
  switch (c) {
    case KernelCategory::UNARY_ELEMENTWISE: return "UNARY_ELEMENTWISE";
    case KernelCategory::BINARY_ELEMENTWISE: return "BINARY_ELEMENTWISE";
    case KernelCategory::TERNARY_ELEMENTWISE: return "TERNARY_ELEMENTWISE";
    case KernelCategory::COMPARISON: return "COMPARISON";
    case KernelCategory::LOGICAL: return "LOGICAL";
    case KernelCategory::REDUCTION: return "REDUCTION";
    case KernelCategory::IDENTITY: return "IDENTITY";
  }
  return "?";
}

namespace {
bool categoryNeedsBody(KernelCategory c) { return c != KernelCategory::REDUCTION; }
}  // namespace

std::string KernelSpec::validate() const {
  if (name.empty()) return "spec has no name";
  for (const auto& a : aliases)
    if (a.empty()) return name + ": empty alias";
  if (numInputs < 0) return name + ": negative numInputs";
  if (dtypes == 0) return name + ": empty dtype mask";

  std::set<int> tArgIndices;
  for (const auto& s : scalars) {
    if (s.name.empty()) return name + ": scalar param with empty name";
    if (s.tArgIndex < 0) return name + ": scalar param '" + s.name + "' has negative tArg index";
    if (!tArgIndices.insert(s.tArgIndex).second)
      return name + ": duplicate tArg index " + std::to_string(s.tArgIndex);
  }
  const auto numScalars = static_cast<int32_t>(scalars.size());

  if (categoryNeedsBody(category)) {
    if (!hasBody) return name + ": category " + kernelCategoryName(category) + " requires a body";
    if (reduction.present) return name + ": non-reduction spec must not carry a reduction triple";
    auto err = body.validate();
    if (!err.empty()) return name + ": body: " + err;
    if (body.inputArity() != numInputs)
      return name + ": body references " + std::to_string(body.inputArity()) +
             " inputs but numInputs is " + std::to_string(numInputs);
    if (body.scalarArity() > numScalars)
      return name + ": body references scalar param " + std::to_string(body.scalarArity() - 1) +
             " but only " + std::to_string(numScalars) + " scalar params are declared";
    return "";
  }

  // REDUCTION
  if (hasBody) return name + ": reduction spec must not carry an elementwise body";
  if (!reduction.present) return name + ": reduction spec requires an (init, combine, finalize) triple";

  auto err = reduction.init.validate();
  if (!err.empty()) return name + ": reduction init: " + err;
  if (reduction.init.inputArity() != 0)
    return name + ": reduction init must not reference inputs";

  err = reduction.combine.validate();
  if (!err.empty()) return name + ": reduction combine: " + err;
  if (reduction.combine.inputArity() != 2)
    return name + ": reduction combine must reference exactly INPUT(0)=accumulator and INPUT(1)=element";

  err = reduction.finalize.validate();
  if (!err.empty()) return name + ": reduction finalize: " + err;
  if (reduction.finalize.inputArity() < 1 || reduction.finalize.inputArity() > 2)
    return name + ": reduction finalize must reference INPUT(0)=accumulator (and optionally INPUT(1)=count)";

  for (const ExprGraph* g : {&reduction.init, &reduction.combine, &reduction.finalize})
    if (g->scalarArity() > numScalars)
      return name + ": reduction triple references an undeclared scalar param";

  return "";
}

// ── registry ────────────────────────────────────────────────────────────────

struct KernelSpecRegistry::Impl {
  mutable std::mutex mutex;
  std::map<std::string, KernelSpec> specs;          // primary name -> spec
  std::map<std::string, std::string> aliasToName;   // alias -> primary name
  std::map<LongType, std::string> hashToName;       // canonical descriptor hash -> primary name
};

KernelSpecRegistry& KernelSpecRegistry::getInstance() {
  static KernelSpecRegistry instance;
  return instance;
}

KernelSpecRegistry::Impl& KernelSpecRegistry::impl() const {
  static Impl instance;
  return instance;
}

bool KernelSpecRegistry::add(const KernelSpec& spec, std::string* error) {
  KernelSpec resolved = spec;
  auto* op = sd::ops::OpRegistrator::getInstance().getOperation(spec.name.c_str());
  if (op == nullptr || op->getOpDescriptor() == nullptr) {
    if (error) *error = spec.name + ": no canonical registered operation";
    return false;
  }
  auto* descriptor = op->getOpDescriptor();
  resolved.descriptorHash = descriptor->getHash();
  const uint64_t descriptorTraits = descriptor->getTraits64();
  if ((descriptorTraits & resolved.traits) != resolved.traits) {
    if (error) *error = spec.name + ": KernelSpec traits are not an op-local descriptor subset";
    return false;
  }
  auto err = resolved.validate();
  if (!err.empty()) {
    if (error) *error = err;
    return false;
  }
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.specs.count(resolved.name) || state.aliasToName.count(resolved.name)) {
    if (error) *error = spec.name + ": name already registered";
    return false;
  }
  if (state.hashToName.count(resolved.descriptorHash)) {
    if (error) *error = resolved.name + ": descriptor hash already registered";
    return false;
  }
  for (const auto& a : spec.aliases) {
    if (state.specs.count(a) || state.aliasToName.count(a)) {
      if (error) *error = spec.name + ": alias '" + a + "' already registered";
      return false;
    }
  }
  for (const auto& a : resolved.aliases) state.aliasToName[a] = resolved.name;
  state.hashToName[resolved.descriptorHash] = resolved.name;
  state.specs[resolved.name] = resolved;
  return true;
}

const KernelSpec* KernelSpecRegistry::find(const std::string& nameOrAlias) const {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.specs.find(nameOrAlias);
  if (it != state.specs.end()) return &it->second;
  auto alias = state.aliasToName.find(nameOrAlias);
  if (alias != state.aliasToName.end()) {
    auto primary = state.specs.find(alias->second);
    if (primary != state.specs.end()) return &primary->second;
  }
  return nullptr;
}

const KernelSpec* KernelSpecRegistry::find(LongType descriptorHash) const {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto hash = state.hashToName.find(descriptorHash);
  if (hash == state.hashToName.end()) return nullptr;
  auto spec = state.specs.find(hash->second);
  return spec == state.specs.end() ? nullptr : &spec->second;
}

std::vector<const KernelSpec*> KernelSpecRegistry::all() const {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mutex);
  std::vector<const KernelSpec*> result;
  result.reserve(state.specs.size());
  for (const auto& entry : state.specs) result.push_back(&entry.second);
  return result;
}

size_t KernelSpecRegistry::size() const {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.specs.size();
}

// ── builder ─────────────────────────────────────────────────────────────────

namespace {
int defaultInputsFor(KernelCategory c) {
  switch (c) {
    case KernelCategory::UNARY_ELEMENTWISE:
    case KernelCategory::IDENTITY:
    case KernelCategory::REDUCTION:
      return 1;
    case KernelCategory::BINARY_ELEMENTWISE:
    case KernelCategory::COMPARISON:
    case KernelCategory::LOGICAL:
      return 2;
    case KernelCategory::TERNARY_ELEMENTWISE:
      return 3;
  }
  return 1;
}
}  // namespace

KernelSpecBuilder::KernelSpecBuilder(std::string name) { spec_.name = std::move(name); }

KernelSpecBuilder& KernelSpecBuilder::alias(std::string a) {
  spec_.aliases.push_back(std::move(a));
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::category(KernelCategory c) {
  spec_.category = c;
  spec_.numInputs = defaultInputsFor(c);
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::traits(uint64_t traitMask) {
  spec_.traits = traitMask;
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::dtypes(uint32_t dtypeMask) {
  spec_.dtypes = dtypeMask;
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::inputs(int n) {
  spec_.numInputs = n;
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::scalar(std::string name, int tArgIndex, double defaultValue) {
  ScalarParamSpec s;
  s.name = std::move(name);
  s.tArgIndex = tArgIndex;
  s.defaultValue = defaultValue;
  spec_.scalars.push_back(std::move(s));
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::body(const std::function<Expr(ExprGraph&)>& build) {
  spec_.body = ExprGraph();
  Expr root = build(spec_.body);
  spec_.body.setRoot(root);
  spec_.hasBody = true;
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::reduction(const std::function<Expr(ExprGraph&)>& init,
                                                const std::function<Expr(ExprGraph&)>& combine,
                                                const std::function<Expr(ExprGraph&)>& finalize) {
  spec_.reduction = ReductionTripleSpec();
  spec_.reduction.init.setRoot(init(spec_.reduction.init));
  spec_.reduction.combine.setRoot(combine(spec_.reduction.combine));
  spec_.reduction.finalize.setRoot(finalize(spec_.reduction.finalize));
  spec_.reduction.present = true;
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::handWritten(std::string engine) {
  spec_.handWrittenOverride.push_back(std::move(engine));
  return *this;
}

KernelSpecBuilder& KernelSpecBuilder::notes(std::string text) {
  spec_.notes = std::move(text);
  return *this;
}

bool KernelSpecBuilder::registerSpec(std::string* error) {
  return KernelSpecRegistry::getInstance().add(spec_, error);
}

KernelSpec KernelSpecBuilder::build() const { return spec_; }

}  // namespace kernelspec
}  // namespace sd
